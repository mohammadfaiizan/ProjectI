# Neuromorphic Computing and Spiking Neural Networks

## Table of Contents

1. [Introduction](#introduction)
2. [Biological Inspiration](#biological-inspiration)
3. [Spiking Neural Networks](#spiking-neural-networks)
4. [Leaky Integrate-and-Fire Neuron Model](#leaky-integrate-and-fire-neuron-model)
5. [Spike-Timing-Dependent Plasticity (STDP)](#spike-timing-dependent-plasticity-stdp)
6. [Neuromorphic Hardware](#neuromorphic-hardware)
7. [Event-Driven Processing](#event-driven-processing)
8. [Temporal Coding](#temporal-coding)
9. [Energy Efficiency and Applications](#energy-efficiency-and-applications)
10. [Key Takeaways](#key-takeaways)

## Introduction

Neuromorphic computing aims to design computing systems inspired by the brain's architecture and principles. Spiking neural networks (SNNs) are a key component, using discrete spikes (action potentials) for communication, similar to biological neurons.

Unlike traditional artificial neural networks that use continuous activations, SNNs operate on discrete events, enabling event-driven computation that can be highly energy-efficient. Neuromorphic hardware implements SNNs efficiently, potentially offering orders of magnitude improvement in energy consumption for certain applications.

Key research directions:
- How to design efficient SNN architectures?
- How to train SNNs effectively?
- How to implement SNNs on neuromorphic hardware?
- What applications benefit from neuromorphic computing?

## Biological Inspiration

Understanding biological neural networks provides inspiration for neuromorphic systems.

### Biological Neurons

**Structure**:
- **Dendrites**: Receive inputs
- **Soma (cell body)**: Integrates inputs
- **Axon**: Transmits outputs
- **Synapses**: Connections between neurons

**Function**:
- Receive inputs from other neurons
- Integrate inputs over time
- Generate action potential (spike) when threshold exceeded
- Transmit spike to connected neurons

### Action Potentials

**Spike**: Brief electrical pulse (~1ms duration)
**All-or-nothing**: Binary event (spike or no spike)
**Frequency coding**: Information encoded in spike rate
**Temporal coding**: Information encoded in spike timing

### Synaptic Plasticity

**Hebbian learning**: "Neurons that fire together, wire together"
**Long-term potentiation (LTP)**: Strengthening of synapses
**Long-term depression (LTD)**: Weakening of synapses
**Spike-timing-dependent plasticity (STDP)**: Depends on relative spike timing

### Brain Properties

**Massive parallelism**: ~10^11 neurons, ~10^15 synapses
**Low power**: ~20W for entire brain
**Fault tolerance**: Robust to neuron death
**Adaptability**: Continual learning and adaptation

## Spiking Neural Networks

SNNs are neural networks that use discrete spikes for communication and computation.

### Key Differences from ANNs

| Property | ANN | SNN |
|----------|-----|-----|
| Activation | Continuous | Discrete (spikes) |
| Time | Static | Temporal dynamics |
| Computation | Synchronous | Event-driven |
| Energy | High | Potentially low |

### Spiking Neuron Models

**Integrate-and-Fire (IF)**: Simplest model
**Leaky Integrate-and-Fire (LIF)**: Includes leakage
**Izhikevich**: More biologically realistic
**Hodgkin-Huxley**: Detailed biophysical model

### Network Architecture

**Layers**: Similar to ANNs (input, hidden, output)
**Connections**: Weighted synapses
**Dynamics**: Temporal evolution of membrane potentials

### Information Encoding

**Rate coding**: Information in spike rate (Hz)
**Temporal coding**: Information in spike timing
**Population coding**: Information across neuron population
**Latency coding**: Information in time to first spike

## Leaky Integrate-and-Fire Neuron Model

The LIF model is widely used in neuromorphic computing for its balance of biological realism and computational efficiency.

### Dynamics

**Membrane potential**: $V(t)$ evolves according to:

$$\tau_m \frac{dV}{dt} = -(V(t) - V_{rest}) + R_m I(t)$$

where:
- $\tau_m = R_m C_m$: Membrane time constant
- $V_{rest}$: Resting potential
- $R_m$: Membrane resistance
- $C_m$: Membrane capacitance
- $I(t)$: Input current

### Spike Generation

**Threshold**: $V_{th}$ (threshold potential)
**Spike condition**: If $V(t) \geq V_{th}$:
- Emit spike
- Reset: $V(t) \leftarrow V_{reset}$
- Refractory period: $V$ clamped for short time

### Discrete-Time Implementation

**Euler method**:
$$V[t+1] = V[t] + \frac{\Delta t}{\tau_m}(-(V[t] - V_{rest}) + R_m I[t])$$

**Spike**: $s[t] = 1$ if $V[t] \geq V_{th}$, else $s[t] = 0$
**Reset**: If $s[t] = 1$, then $V[t+1] = V_{reset}$

### Synaptic Input

**Input current**: 
$$I[t] = \sum_j w_j s_j[t]$$

where $w_j$ are synaptic weights and $s_j[t]$ are input spikes.

### Parameters

**Time constant** $\tau_m$: Controls integration time (~10-20ms)
**Threshold** $V_{th}$: Determines spiking sensitivity
**Reset** $V_{reset}$: Potential after spike
**Refractory period**: Prevents immediate re-spiking

## Spike-Timing-Dependent Plasticity (STDP)

STDP is a learning rule based on the relative timing of pre- and post-synaptic spikes.

### Basic Rule

**LTP (Long-Term Potentiation)**: If pre-synaptic spike before post-synaptic spike, strengthen synapse
**LTD (Long-Term Depression)**: If post-synaptic spike before pre-synaptic spike, weaken synapse

### Mathematical Formulation

**Weight update**:
$$\Delta w = \begin{cases}
A_+ e^{-\Delta t / \tau_+} & \text{if } \Delta t > 0 \text{ (LTP)} \\
-A_- e^{\Delta t / \tau_-} & \text{if } \Delta t < 0 \text{ (LTD)}
\end{cases}$$

where $\Delta t = t_{post} - t_{pre}$ is the time difference.

**Parameters**:
- $A_+$, $A_-$: Learning rates
- $\tau_+$, $\tau_-$: Time constants (~10-20ms)

### Window Function

**Typical window**:
- LTP window: ~20ms (pre before post)
- LTD window: ~20ms (post before pre)
- Decay: Exponential decay with distance from zero

### Properties

**Causality**: Strengthens causal connections
**Temporal precision**: Sensitive to millisecond timing
**Unsupervised**: No external labels needed
**Local**: Only depends on local spike times

### Limitations

**Stability**: May lead to runaway potentiation or depression
**Rate dependence**: Also depends on firing rates
**Multiple spikes**: Complex with multiple spikes

### Variants

**Triplet STDP**: Considers triplets of spikes
**Voltage-dependent STDP**: Depends on post-synaptic voltage
**Calcium-based**: Uses calcium concentration

## Neuromorphic Hardware

Neuromorphic hardware implements SNNs efficiently using specialized architectures.

### Design Principles

**Event-driven**: Process only when events occur
**Massive parallelism**: Many neurons operate simultaneously
**Local memory**: Synaptic weights stored locally
**Low precision**: Can use low-bit weights

### Intel Loihi

**Architecture**: 
- 128 neuromorphic cores per chip
- ~130,000 neurons, ~130 million synapses per chip
- On-chip learning (STDP)
- Asynchronous, event-driven

**Features**:
- Programmable neuron models
- Synaptic delays
- Hierarchical connectivity
- Mesh interconnect

**Applications**: Pattern recognition, optimization, robotics

### IBM TrueNorth

**Architecture**:
- 4096 neurosynaptic cores
- 1 million neurons, 256 million synapses
- Fixed LIF neuron model
- Event-driven communication

**Features**:
- Low power (~70mW)
- Deterministic operation
- Scalable to multiple chips

**Applications**: Real-time pattern recognition, sensor processing

### SpiNNaker

**Architecture**:
- ARM processors as neurons
- Software-based SNN simulation
- Real-time simulation
- Large-scale (million+ neurons)

**Features**:
- Flexible neuron models
- Configurable connectivity
- Real-time operation

**Applications**: Brain simulation, robotics

### Other Platforms

**BrainScaleS**: Mixed-signal (analog neurons, digital synapses)
**Neurogrid**: Analog VLSI implementation
**Dynap-SE**: Dynamic Neuromorphic Asynchronous Processor

### Comparison

| Platform | Neurons | Synapses | Power | Learning |
|----------|---------|----------|-------|----------|
| Loihi | 130K | 130M | ~1W | On-chip |
| TrueNorth | 1M | 256M | ~70mW | Off-chip |
| SpiNNaker | 1M+ | Flexible | ~1W | Software |

## Event-Driven Processing

Event-driven processing computes only when events (spikes) occur, enabling energy efficiency.

### Principle

**Traditional**: Process all data at fixed intervals
**Event-driven**: Process only when events occur
**Sparsity**: Exploits sparsity in spike patterns

### Advantages

**Energy efficiency**: No computation when no spikes
**Latency**: Low latency (responds immediately to events)
**Scalability**: Scales with activity, not data size

### Implementation

**Event queue**: Queue of pending spikes
**Processing**: Process spikes as they arrive
**Scheduling**: Efficient scheduling of spike processing

### Challenges

**Synchronization**: Handling asynchronous events
**Load balancing**: Distributing load across processors
**Memory**: Managing event queues

### Applications

**Sensor processing**: Process only when sensors detect changes
**Robotics**: React to events in real-time
**Edge computing**: Low-power edge devices

## Temporal Coding

Temporal coding uses spike timing to encode information, not just spike rates.

### Rate vs Temporal Coding

**Rate coding**: Information in average firing rate
- Simple, robust
- Requires averaging over time
- Less efficient

**Temporal coding**: Information in precise spike timing
- More efficient
- Higher information capacity
- Requires precise timing

### Time-to-First-Spike

**Latency coding**: Time to first spike encodes stimulus strength
- Stronger stimulus → earlier spike
- Single spike can encode information
- Very efficient

### Spike Patterns

**Precise timing**: Relative timing of spikes encodes information
**Synchrony**: Synchronized spikes encode correlated features
**Sequences**: Temporal sequences encode temporal patterns

### Advantages

**Efficiency**: Can encode information with few spikes
**Speed**: Can respond quickly (first spike)
**Precision**: High information content per spike

### Challenges

**Noise**: Sensitive to noise and jitter
**Implementation**: Requires precise timing
**Learning**: Harder to learn temporal patterns

## Energy Efficiency and Applications

Neuromorphic computing offers potential for orders of magnitude improvement in energy efficiency.

### Energy Consumption

**Traditional**: ~100 pJ per operation
**Neuromorphic**: ~1-10 pJ per spike (potential)
**Brain**: ~10 fJ per spike (biological)

**Factors**:
- Event-driven: Only process when needed
- Low precision: Can use low-bit operations
- Local memory: Reduces data movement

### Applications

**Edge AI**: Low-power edge devices
- Sensor processing
- Always-on applications
- Battery-powered devices

**Robotics**: Real-time control
- Sensorimotor control
- Autonomous navigation
- Manipulation

**Neuromorphic sensors**: Event cameras (DVS)
- High temporal resolution
- Low latency
- Low power

**Pattern recognition**: Real-time classification
- Object recognition
- Speech recognition
- Gesture recognition

### Benchmarking

**Tasks**: Image classification, gesture recognition, etc.
**Metrics**: Accuracy, latency, energy
**Comparison**: vs traditional ANNs on CPUs/GPUs

### Challenges

**Training**: Harder to train SNNs
**Accuracy**: May have lower accuracy than ANNs
**Hardware**: Limited availability
**Software**: Less mature tooling

## Key Takeaways

1. **Neuromorphic computing** designs systems inspired by the brain, using spiking neural networks for event-driven, energy-efficient computation.

2. **Spiking neural networks** use discrete spikes for communication, enabling temporal dynamics and event-driven processing unlike traditional ANNs.

3. **Leaky Integrate-and-Fire (LIF)** model balances biological realism and computational efficiency, widely used in neuromorphic systems.

4. **Spike-Timing-Dependent Plasticity (STDP)** is an unsupervised learning rule based on relative spike timing, strengthening causal connections.

5. **Neuromorphic hardware** (Loihi, TrueNorth, SpiNNaker) implements SNNs efficiently with event-driven processing and massive parallelism.

6. **Event-driven processing** computes only when spikes occur, enabling energy efficiency by exploiting sparsity.

7. **Temporal coding** uses spike timing to encode information, potentially more efficient than rate coding.

8. **Energy efficiency** is a key advantage, with potential for orders of magnitude improvement over traditional computing.

9. **Applications** include edge AI, robotics, neuromorphic sensors, and real-time pattern recognition.

10. **Future directions** include improving training methods, developing better hardware, expanding applications, and bridging the gap between SNNs and ANNs.

## References

- Maass, W. (1997). "Networks of Spiking Neurons: The Third Generation of Neural Network Models." Neural Networks 10, 1659-1671
- Gerstner, W., & Kistler, W. M. (2002). "Spiking Neuron Models: Single Neurons, Populations, Plasticity." Cambridge University Press
- Davies, M., et al. (2018). "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning." IEEE Micro 38, 82-99
- Merolla, P. A., et al. (2014). "A Million Spiking-Neuron Integrated Circuit with a Scalable Communication Network and Interface." Science 345, 668-673
- Furber, S. B., et al. (2014). "The SpiNNaker Project." Proceedings of the IEEE 102, 652-665
- Bi, G.-q., & Poo, M.-m. (1998). "Synaptic Modifications in Cultured Hippocampal Neurons: Dependence on Spike Timing, Synaptic Strength, and Postsynaptic Cell Type." Journal of Neuroscience 18, 10464-10472
- Roy, K., et al. (2019). "Towards Spike-Based Machine Intelligence with Neuromorphic Computing." Nature 575, 607-617
- Pfeiffer, M., & Pfeil, T. (2018). "Deep Learning with Spiking Neurons: Opportunities and Challenges." Frontiers in Neuroscience 12, 774
- Diehl, P. U., & Cook, M. (2015). "Unsupervised Learning of Digit Recognition Using Spike-Timing-Dependent Plasticity." Frontiers in Computational Neuroscience 9, 99
- Lobo, J. L., et al. (2020). "Spiking Neural Networks and Online Learning: An Overview and Perspectives." Neural Networks 121, 88-100
