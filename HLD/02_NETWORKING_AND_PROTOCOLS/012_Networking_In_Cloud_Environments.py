"""
NETWORKING IN CLOUD ENVIRONMENTS
===================================

Problem Statement:
Cloud deployments require understanding virtual networking: VPCs, subnets,
routing, security groups, NAT gateways, peering, and private connectivity.
Misconfigurations lead to data exposure, unreachable services, or excessive
egress costs.

AWS Networking Building Blocks:
  VPC (Virtual Private Cloud): isolated network for your resources
  Subnets: subdivide VPC across AZs (public vs private)
  Internet Gateway: enables VPC to reach the internet
  NAT Gateway: lets private subnet reach internet (outbound only)
  Security Groups: stateful firewall per resource (instance level)
  NACLs: stateless firewall per subnet (subnet level)
  Route Tables: control traffic flow between subnets and gateways
  VPC Peering: connect two VPCs (no transitive routing)
  Transit Gateway: hub-and-spoke to connect many VPCs
  PrivateLink: access AWS services without internet (private IP)

3-Tier Architecture in VPC:
  Public Subnet  : Load balancer (internet-facing)
  Private Subnet : Application servers (no public IP)
  DB Subnet      : Databases (no internet route whatsoever)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import ipaddress


class SubnetType(Enum):
    PUBLIC   = "public"    # has internet gateway route
    PRIVATE  = "private"   # outbound via NAT gateway
    ISOLATED = "isolated"  # no internet route at all (DB tier)


class TrafficDirection(Enum):
    INBOUND  = "inbound"
    OUTBOUND = "outbound"


@dataclass
class Subnet:
    subnet_id   : str
    cidr        : str
    subnet_type : SubnetType
    az          : str   # availability zone, e.g. "us-east-1a"
    resources   : List[str] = field(default_factory=list)

    @property
    def network(self) -> ipaddress.IPv4Network:
        return ipaddress.ip_network(self.cidr, strict=False)

    def add_resource(self, resource_id: str):
        self.resources.append(resource_id)


@dataclass
class RouteEntry:
    destination_cidr : str
    target           : str   # "igw-xxx" | "nat-xxx" | "local" | "tgw-xxx"
    description      : str = ""

    def matches(self, dst_ip: str) -> bool:
        try:
            network = ipaddress.ip_network(self.destination_cidr, strict=False)
            return ipaddress.ip_address(dst_ip) in network
        except ValueError:
            return False


@dataclass
class SecurityGroupRule:
    direction  : TrafficDirection
    protocol   : str       # tcp/udp/icmp/-1(all)
    port_from  : int
    port_to    : int
    source_cidr: str       # 0.0.0.0/0 or specific CIDR or sg-id
    description: str = ""

    def allows(self, direction: TrafficDirection, port: int, src_ip: str) -> bool:
        if self.direction != direction:
            return False
        if self.port_from > port or self.port_to < port:
            return False
        if self.source_cidr == "0.0.0.0/0":
            return True
        try:
            return ipaddress.ip_address(src_ip) in ipaddress.ip_network(self.source_cidr, strict=False)
        except ValueError:
            return True   # sg reference — allow


# ─────────────────────────────────────────────
# VPC
# ─────────────────────────────────────────────

class VPC:
    def __init__(self, vpc_id: str, cidr: str, region: str):
        self.vpc_id  = vpc_id
        self.cidr    = cidr
        self.region  = region
        self.subnets : Dict[str, Subnet] = {}

    def add_subnet(self, subnet: Subnet):
        self.subnets[subnet.subnet_id] = subnet

    def find_subnet_for_ip(self, ip: str) -> Optional[Subnet]:
        for subnet in self.subnets.values():
            if ipaddress.ip_address(ip) in subnet.network:
                return subnet
        return None


# ─────────────────────────────────────────────
# ROUTE TABLE
# ─────────────────────────────────────────────

class RouteTable:
    def __init__(self, rt_id: str):
        self.rt_id   = rt_id
        self.routes  : List[RouteEntry] = []

    def add_route(self, entry: RouteEntry):
        self.routes.append(entry)

    def lookup(self, dst_ip: str) -> Optional[RouteEntry]:
        """Longest prefix match (most specific wins)."""
        matches = [r for r in self.routes if r.matches(dst_ip)]
        if not matches:
            return None
        return max(matches, key=lambda r: ipaddress.ip_network(r.destination_cidr).prefixlen)

    def show(self, name: str = ""):
        print(f"\n  Route Table [{self.rt_id}] {name}:")
        print(f"  {'Destination':<20} {'Target':<20} Description")
        print(f"  {'─'*60}")
        for r in self.routes:
            print(f"  {r.destination_cidr:<20} {r.target:<20} {r.description}")


# ─────────────────────────────────────────────
# SECURITY GROUP
# ─────────────────────────────────────────────

class SecurityGroup:
    def __init__(self, sg_id: str, name: str):
        self.sg_id = sg_id
        self.name  = name
        self.rules : List[SecurityGroupRule] = []

    def add_rule(self, rule: SecurityGroupRule):
        self.rules.append(rule)

    def evaluate(self, direction: TrafficDirection, port: int, src_ip: str) -> bool:
        return any(r.allows(direction, port, src_ip) for r in self.rules)

    def show(self):
        print(f"\n  Security Group [{self.sg_id}] {self.name}:")
        for r in self.rules:
            print(f"    {r.direction.value:<10} {r.protocol:<5} "
                  f"{r.port_from}-{r.port_to:<6} from {r.source_cidr:<20} {r.description}")


# ─────────────────────────────────────────────
# NAT GATEWAY
# ─────────────────────────────────────────────

class NATGateway:
    """
    Allows private subnet resources to initiate outbound internet connections.
    Internet cannot initiate connections to private resources (one-way).
    """

    def __init__(self, nat_id: str, elastic_ip: str):
        self.nat_id     = nat_id
        self.elastic_ip = elastic_ip
        self.bytes_out  = 0
        self.cost_per_gb = 0.045   # USD

    def translate(self, private_ip: str, dst_ip: str) -> str:
        self.bytes_out += 1000   # simulated
        print(f"  NAT: {private_ip} → {self.elastic_ip} → {dst_ip}  (outbound only)")
        return self.elastic_ip

    @property
    def cost_estimate_usd(self) -> float:
        return (self.bytes_out / 1e9) * self.cost_per_gb


# ─────────────────────────────────────────────
# VPC PEERING / TRANSIT GATEWAY
# ─────────────────────────────────────────────

class VPCPeering:
    """Direct peering between two VPCs — no transitive routing."""

    def __init__(self, peering_id: str, vpc_a: str, vpc_b: str,
                 cidr_a: str, cidr_b: str):
        self.peering_id = peering_id
        self.vpc_a      = vpc_a
        self.vpc_b      = vpc_b
        self.cidr_a     = cidr_a
        self.cidr_b     = cidr_b

    def can_route(self, src_vpc: str, dst_ip: str) -> bool:
        if src_vpc == self.vpc_a:
            target_net = ipaddress.ip_network(self.cidr_b, strict=False)
        elif src_vpc == self.vpc_b:
            target_net = ipaddress.ip_network(self.cidr_a, strict=False)
        else:
            return False
        try:
            return ipaddress.ip_address(dst_ip) in target_net
        except ValueError:
            return False


class TransitGateway:
    """
    Hub-and-spoke to connect many VPCs without N*(N-1)/2 peerings.
    Enables transitive routing unlike VPC peering.
    """

    def __init__(self, tgw_id: str):
        self.tgw_id      = tgw_id
        self._attachments: Dict[str, str] = {}   # vpc_id → cidr

    def attach(self, vpc_id: str, cidr: str):
        self._attachments[vpc_id] = cidr
        print(f"  TGW: attached {vpc_id} ({cidr})")

    def route(self, src_vpc: str, dst_ip: str) -> Optional[str]:
        for vpc_id, cidr in self._attachments.items():
            if vpc_id == src_vpc:
                continue
            try:
                if ipaddress.ip_address(dst_ip) in ipaddress.ip_network(cidr, strict=False):
                    return vpc_id
            except ValueError:
                pass
        return None

    def connection_count(self) -> Tuple[int, int]:
        n = len(self._attachments)
        peering_needed = n * (n - 1) // 2
        return n, peering_needed


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_cloud_networking():
    print("=" * 65)
    print("NETWORKING IN CLOUD ENVIRONMENTS")
    print("=" * 65)

    # ── 3-Tier VPC Architecture ───────────────
    print("\n[1] 3-TIER VPC ARCHITECTURE (10.0.0.0/16)")
    print("─" * 55)
    vpc = VPC("vpc-prod", "10.0.0.0/16", "us-east-1")
    subnets = [
        Subnet("sub-pub-1a",  "10.0.1.0/24", SubnetType.PUBLIC,   "us-east-1a"),
        Subnet("sub-pub-1b",  "10.0.2.0/24", SubnetType.PUBLIC,   "us-east-1b"),
        Subnet("sub-priv-1a", "10.0.10.0/24",SubnetType.PRIVATE,  "us-east-1a"),
        Subnet("sub-priv-1b", "10.0.11.0/24",SubnetType.PRIVATE,  "us-east-1b"),
        Subnet("sub-db-1a",   "10.0.20.0/24",SubnetType.ISOLATED, "us-east-1a"),
        Subnet("sub-db-1b",   "10.0.21.0/24",SubnetType.ISOLATED, "us-east-1b"),
    ]
    for s in subnets:
        vpc.add_subnet(s)

    for s in subnets:
        tier = {"public": "ALB / NAT GW", "private": "App servers (ECS/EC2)", "isolated": "RDS / ElastiCache"}
        print(f"  {s.subnet_type.value:<10} {s.subnet_id:<15} {s.cidr:<16} {s.az:<14} {tier[s.subnet_type.value]}")

    # ── Route Tables ──────────────────────────
    print("\n\n[2] ROUTE TABLES")
    print("─" * 55)
    public_rt = RouteTable("rtb-public")
    public_rt.add_route(RouteEntry("10.0.0.0/16", "local",   "VPC local traffic"))
    public_rt.add_route(RouteEntry("0.0.0.0/0",   "igw-001", "Internet Gateway for public subnets"))
    public_rt.show("(public subnets)")

    private_rt = RouteTable("rtb-private")
    private_rt.add_route(RouteEntry("10.0.0.0/16", "local",     "VPC local"))
    private_rt.add_route(RouteEntry("0.0.0.0/0",   "nat-001",   "NAT Gateway for outbound internet"))
    private_rt.show("(private subnets)")

    isolated_rt = RouteTable("rtb-isolated")
    isolated_rt.add_route(RouteEntry("10.0.0.0/16", "local", "VPC local only — no internet"))
    isolated_rt.show("(DB subnets — isolated)")

    # Route lookup examples
    print("\n  Route lookups:")
    for ip, rt_name, rt in [("8.8.8.8", "public", public_rt),
                              ("8.8.8.8", "private", private_rt),
                              ("8.8.8.8", "isolated", isolated_rt),
                              ("10.0.10.5", "public", public_rt)]:
        route = rt.lookup(ip)
        print(f"  [{rt_name}] dst={ip} → {route.target if route else 'no route (drop)'}")

    # ── Security Groups ───────────────────────
    print("\n\n[3] SECURITY GROUPS (stateful)")
    print("─" * 55)
    alb_sg = SecurityGroup("sg-alb", "ALB security group")
    alb_sg.add_rule(SecurityGroupRule(TrafficDirection.INBOUND,  "tcp", 443, 443, "0.0.0.0/0", "HTTPS from internet"))
    alb_sg.add_rule(SecurityGroupRule(TrafficDirection.INBOUND,  "tcp", 80,  80,  "0.0.0.0/0", "HTTP redirect"))
    alb_sg.add_rule(SecurityGroupRule(TrafficDirection.OUTBOUND, "tcp", 0,   65535,"0.0.0.0/0","All outbound"))
    alb_sg.show()

    app_sg = SecurityGroup("sg-app", "App server security group")
    app_sg.add_rule(SecurityGroupRule(TrafficDirection.INBOUND,  "tcp", 8080, 8080, "10.0.1.0/24","ALB only"))
    app_sg.add_rule(SecurityGroupRule(TrafficDirection.INBOUND,  "tcp", 8080, 8080, "10.0.2.0/24","ALB only"))
    app_sg.add_rule(SecurityGroupRule(TrafficDirection.OUTBOUND, "tcp", 0,    65535,"0.0.0.0/0",  "All outbound"))
    app_sg.show()

    db_sg = SecurityGroup("sg-db", "RDS security group")
    db_sg.add_rule(SecurityGroupRule(TrafficDirection.INBOUND,  "tcp", 5432, 5432, "10.0.10.0/24","App subnet only"))
    db_sg.add_rule(SecurityGroupRule(TrafficDirection.INBOUND,  "tcp", 5432, 5432, "10.0.11.0/24","App subnet only"))
    db_sg.show()

    # Evaluate
    print("\n  SG evaluations:")
    cases = [
        (alb_sg, TrafficDirection.INBOUND,  443,  "1.2.3.4",    "external → ALB:443"),
        (alb_sg, TrafficDirection.INBOUND,  22,   "1.2.3.4",    "external → ALB:22 (SSH)"),
        (app_sg, TrafficDirection.INBOUND,  8080, "10.0.1.100", "ALB → App:8080"),
        (app_sg, TrafficDirection.INBOUND,  8080, "5.5.5.5",    "internet → App:8080"),
        (db_sg,  TrafficDirection.INBOUND,  5432, "10.0.10.50", "App → DB:5432"),
        (db_sg,  TrafficDirection.INBOUND,  5432, "1.2.3.4",    "internet → DB:5432"),
    ]
    for sg, direction, port, src_ip, desc in cases:
        allowed = sg.evaluate(direction, port, src_ip)
        icon    = "✅" if allowed else "🚫"
        print(f"  {icon} {desc}")

    # ── NAT Gateway ───────────────────────────
    print("\n\n[4] NAT GATEWAY — PRIVATE SUBNET INTERNET ACCESS")
    print("─" * 55)
    nat = NATGateway("nat-001", "54.210.100.1")
    nat.translate("10.0.10.5",  "142.250.80.46")   # app → google.com
    nat.translate("10.0.10.6",  "52.94.236.248")   # app → s3

    # ── Transit Gateway ───────────────────────
    print("\n\n[5] TRANSIT GATEWAY vs VPC PEERING")
    print("─" * 55)
    tgw = TransitGateway("tgw-001")
    for i, cidr in enumerate(["10.0.0.0/16", "10.1.0.0/16", "10.2.0.0/16",
                                "10.3.0.0/16", "10.4.0.0/16"], 1):
        tgw.attach(f"vpc-{i:03d}", cidr)

    n, peerings = tgw.connection_count()
    print(f"  {n} VPCs attached to Transit Gateway")
    print(f"  Without TGW: would need {peerings} VPC peering connections")
    print(f"  With TGW: {n} attachments only (hub-and-spoke)")

    # Route lookup
    result = tgw.route("vpc-001", "10.2.5.100")
    print(f"  TGW route: vpc-001 → 10.2.5.100 → {result}")

    # ── Best Practices ─────────────────────────
    print("\n\n[6] CLOUD NETWORKING BEST PRACTICES")
    print("─" * 55)
    practices = [
        ("Multi-AZ deployment",      "Subnets in ≥2 AZs for high availability"),
        ("3-tier isolation",          "Public/Private/Isolated subnets for each tier"),
        ("Least-privilege SGs",       "Allow only required ports from required sources"),
        ("No public IPs on app/DB",   "Only ALB in public subnet; app/DB in private"),
        ("NAT for outbound only",     "Private subnets go out via NAT, not IGW"),
        ("VPC Flow Logs",             "Log all accept/reject for security audit"),
        ("Transit Gateway for many",  "Replace N*(N-1)/2 peerings with hub-and-spoke"),
        ("PrivateLink for AWS svcs",  "Access S3/DynamoDB without leaving VPC"),
        ("CIDR planning",             "/16 per env; /24 per subnet; no overlap"),
        ("NACLs as extra layer",      "Subnet-level stateless rules (deny known bad IPs)"),
    ]
    for practice, detail in practices:
        print(f"  • {practice:<30} {detail}")


if __name__ == "__main__":
    demonstrate_cloud_networking()
