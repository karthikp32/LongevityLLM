import re
from datetime import datetime

# Function to read and parse cookies from a raw cookies.txt file (even if not in the correct format)
def read_raw_cookies(cookies_file):
    cookies = []
    with open(cookies_file, 'r') as file:
        raw_data = file.readlines()

    # Regex to capture cookie name and value (basic parsing)
    cookie_pattern = re.compile(r'([^=]+)=([^\n;]+)')

    for line in raw_data:
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and comments
            continue
        match = cookie_pattern.search(line)
        if match:
            name, value = match.groups()
            cookies.append({'name': name.strip(), 'value': value.strip(), 'domain': '.youtube.com', 'path': '/', 'expires': None, 'secure': False})

    return cookies

# Convert cookies to Netscape format
def to_netscape_format(cookie):
    # If no expiration is given, use a default large value for session cookies
    if cookie['expires'] is None:
        expires = int(datetime.now().timestamp()) + 10000000  # 10000000 is a large value for session cookies
    else:
        expires = cookie['expires']
    
    # Ensure domain starts with a period (for Netscape format)
    domain = cookie['domain'].lstrip('.')  # Remove extra period if any
    
    # If cookie is secure, set the appropriate flag
    secure_flag = "TRUE" if cookie['secure'] else "FALSE"
    
    # Format the line in Netscape format
    return f".{domain}    TRUE    {cookie['path']}    {secure_flag}    {expires}    {cookie['name']}    {cookie['value']}"

# Read cookies from raw cookies.txt file
cookies_file = 'cookies.txt'  # Path to your raw cookies file
cookies = read_raw_cookies(cookies_file)

# Write the cookies in Netscape format to a new file
with open('cookies_netscape.txt', 'w') as file:
    for cookie in cookies:
        file.write(to_netscape_format(cookie) + '\n')

print("Cookies have been read and saved in Netscape format.")
