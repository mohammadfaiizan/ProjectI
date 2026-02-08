"""
Sample input module for Code Review Agent.
Contains sample Python code with intentional issues for testing.
"""

from Main import Setup_Review_System, Review_Code


SAMPLE_CODE_1 = """
def process_items(items):
    result = []
    for i in range(len(items)):
        if items[i] > 0:
            result.append(items[i] * 2)
    return result

def find_max_value(numbers):
    max_val = numbers[0]
    for i in range(1, len(numbers)):
        if numbers[i] > max_val:
            max_val = numbers[i]
    return max_val

def calculate_average(values):
    total = 0
    for i in range(len(values)):
        total = total + values[i]
    average = total / len(values)
    return average

def check_user_age(age):
    if age >= 18:
        return "Adult"
    elif age < 18:
        return "Minor"
    else:
        return "Unknown"

def process_data(data_list):
    output = []
    for i in range(len(data_list)):
        item = data_list[i]
        if item is not None:
            processed = item.upper()
            output.append(processed)
    return output

def divide_numbers(a, b):
    result = a / b
    return result

def search_in_list(items, target):
    for i in range(len(items)):
        if items[i] == target:
            return i
    return -1

def remove_duplicates(items):
    result = []
    for i in range(len(items)):
        found = False
        for j in range(len(result)):
            if items[i] == result[j]:
                found = True
                break
        if not found:
            result.append(items[i])
    return result

def count_occurrences(items, target):
    count = 0
    for i in range(len(items)):
        if items[i] == target:
            count = count + 1
    return count

def reverse_list(items):
    result = []
    for i in range(len(items) - 1, -1, -1):
        result.append(items[i])
    return result

def get_last_element(items):
    return items[len(items)]
"""


SAMPLE_CODE_2 = """
import os
import subprocess
import pickle

def execute_user_command(command):
    result = os.system(command)
    return result

def authenticate_user(username, password):
    api_key = "sk-1234567890abcdef"
    secret_token = "my_secret_token_12345"
    if username == "admin" and password == "admin123":
        return True
    return False

def query_database(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = execute_query(query)
    return result

def process_user_input(user_data):
    code = user_data.get("code")
    result = eval(code)
    return result

def download_file(url):
    command = f"wget {url}"
    subprocess.call(command, shell=True)

def store_credentials(username, password):
    with open("credentials.txt", "w") as f:
        f.write(f"Username: {username}\n")
        f.write(f"Password: {password}\n")

def validate_input(user_input):
    if len(user_input) > 0:
        return True
    return False

def get_user_data(user_id):
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    return execute_query(query)

def execute_sql_query(query_string):
    sql = "SELECT * FROM table WHERE condition = '" + query_string + "'"
    return execute_query(sql)

def deserialize_data(data):
    return pickle.loads(data)

def run_system_command(cmd):
    os.system(cmd)

def process_file_path(user_path):
    file_path = "/data/" + user_path
    with open(file_path, "r") as f:
        return f.read()

def get_api_response(endpoint):
    api_key = "hardcoded_api_key_xyz123"
    url = endpoint + "?key=" + api_key
    return fetch_url(url)

def save_user_session(session_data):
    with open("sessions.pkl", "wb") as f:
        pickle.dump(session_data, f)
"""


SAMPLE_CODE_3 = """
def calc(x,y):
    z=x+y
    return z

def processdata(data):
    result=[]
    for i in range(len(data)):
        if data[i]>10:
            result.append(data[i]*2)
        elif data[i]>5:
            result.append(data[i]*1.5)
        else:
            result.append(data[i])
    return result

def fn1(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z):
    return a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w+x+y+z

def check(x):
    if x==1:
        return "one"
    elif x==2:
        return "two"
    elif x==3:
        return "three"
    elif x==4:
        return "four"
    elif x==5:
        return "five"
    elif x==6:
        return "six"
    elif x==7:
        return "seven"
    elif x==8:
        return "eight"
    elif x==9:
        return "nine"
    else:
        return "other"

def calculate(x):
    return x*3.14159*2.71828

def process(items):
    output=[]
    for item in items:
        if item.status==True:
            if item.value>100:
                if item.category=="A":
                    output.append(item.value*1.1)
                elif item.category=="B":
                    output.append(item.value*1.2)
                else:
                    output.append(item.value)
            else:
                output.append(item.value*0.9)
        else:
            output.append(0)
    return output

def dothing(x):
    y=x*2
    z=y+5
    w=z*3
    return w

def proc(items):
    out=[]
    for i in items:
        if i>10:
            out.append(i*2)
    return out

def calc2(a,b):
    return a+b*2-3+4*5-6

def handle(data):
    result=[]
    for d in data:
        if d==True:
            result.append(1)
        elif d==False:
            result.append(0)
        else:
            result.append(-1)
    return result

def transform(input):
    output=input*2+10-5*3
    return output

def validate(val):
    if val==True:
        return True
    elif val==False:
        return False
    else:
        return None

def compute(x,y,z):
    temp=x+y
    result=temp*z
    final=result/2
    return final

def process_items(items):
    output=[]
    for item in items:
        if item>0:
            if item<10:
                output.append(item*1)
            elif item<20:
                output.append(item*2)
            elif item<30:
                output.append(item*3)
            else:
                output.append(item*4)
        else:
            output.append(0)
    return output
"""


def Run_Samples():
    """
    Run code review on all sample code snippets and print detailed reports.
    """
    print("=" * 80)
    print("CODE REVIEW AGENT - SAMPLE CODE REVIEW")
    print("=" * 80)
    
    try:
        review_system = Setup_Review_System()
    except Exception as e:
        print(f"Error setting up review system: {str(e)}")
        print("Make sure OPENAI_API_KEY environment variable is set.")
        return
    
    samples = [
        ("Sample Code 1 - Bug Issues", SAMPLE_CODE_1),
        ("Sample Code 2 - Security Issues", SAMPLE_CODE_2),
        ("Sample Code 3 - Style Issues", SAMPLE_CODE_3)
    ]
    
    for sample_name, sample_code in samples:
        print("\n" + "=" * 80)
        print(f"REVIEWING: {sample_name}")
        print("=" * 80)
        
        print("\nCode:")
        print("-" * 80)
        print(sample_code)
        print("-" * 80)
        
        try:
            result = Review_Code(sample_code, review_system)
            
            print(f"\nReview completed for {sample_name}")
            print(f"Overall Score: {result.get('overall_score', 0):.1f}/100")
            
            bug_count = len(result.get("bug_issues", []))
            security_count = len(result.get("security_issues", []))
            style_count = len(result.get("style_issues", []))
            performance_count = len(result.get("performance_issues", []))
            
            print(f"Issues found: Bugs={bug_count}, Security={security_count}, "
                  f"Style={style_count}, Performance={performance_count}")
            
        except Exception as e:
            print(f"Error reviewing {sample_name}: {str(e)}")
        
        print("\n" + "-" * 80)
    
    print("\n" + "=" * 80)
    print("ALL SAMPLES REVIEWED")
    print("=" * 80)


if __name__ == "__main__":
    Run_Samples()
