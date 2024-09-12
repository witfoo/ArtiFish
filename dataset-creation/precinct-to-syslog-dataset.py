## Connects to a WitFoo Precinct Cassandra cluster, analyzes Incidents and Artifacts to create a dataset for training how to translate syslog formats to English.
import datetime
import csv
import os
import random
import json
from datasets import load_dataset
import pandas as pd
from datasets import Dataset
import ssl
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement

# Update the following variables with the correct values
cassandra_port = 9042
username = "cassandra"
password = "password"
cassandra_host = "witfoo.acme.local"
records_per_streamName = 1000 # Number of records to generate per streamName
max_records = 10000000 # Maximum number of records to generate

# Dictionaries to store the fake data
fake_orgs = ["acme", "umbrella", "stark", "wayne", "oscorp"]
fake_domain_names = ["acme.com", "umbrella.com", "stark.com", "wayne.com", "oscorp.com"]
fake_usernames = ["johndoe", "janedoe", "alice", "bob", "charlie", "fish", "nighthawk"]
protected_strings = []
streamNames = []
products = []

start_time = datetime.datetime.now()
def generate_fake_ipv4():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def sanitize(row):
    global protected_strings, products, streamNames
    # JSON decode the row
    json_obj = json.loads(row)
    # If the message is not present, or is set to None skip the row
    if "message" not in json_obj or not json_obj["message"]:
        return False
    # If Organization is present, replace it with a fake organization
    if "streamname" in json_obj:
        json_obj["streamName"] = json_obj["streamname"]
        del json_obj["streamname"]
    if "clientip" in json_obj:
        json_obj["clientIP"] = json_obj["clientip"]
        del json_obj["clientip"]
    if "serverip" in json_obj:
        json_obj["serverIP"] = json_obj["serverip"]
        del json_obj["serverip"]
    if "username" in json_obj:
        json_obj["userName"] = json_obj["username"]
        del json_obj["username"]
    if "organization" in json_obj:
        # Don't replace if the length of the organization is less than 5
        if len(json_obj["organization"]) > 1:
            if json_obj["organization"] not in protected_strings:
                protected_strings.append(json_obj["organization"])            
    if "userName" in json_obj:
        if (json_obj["userName"]):
            if len(json_obj["userName"]) > 1:
                old = json_obj["userName"]
                json_obj["userName"] = random.choice(fake_usernames)
                # Replace the old userName with the new userName in the message
                json_obj["message"] = json_obj["message"].replace(old, json_obj["userName"])
    if "clientIP" in json_obj:
        if (json_obj["clientIP"]):
            if len(json_obj["clientIP"]) > 1:
                old = json_obj["clientIP"]
                json_obj["clientIP"] = generate_fake_ipv4()
                # Replace the old clientIP with the new clientIP in the message
                json_obj["message"] = json_obj["message"].replace(old, json_obj["clientIP"])
    if "serverIP" in json_obj:
        if (json_obj["serverIP"]):
            if len(json_obj["serverIP"]) > 1:
                old = json_obj["serverIP"]
                json_obj["serverIP"] = generate_fake_ipv4()
                # Replace the old serverIP with the new serverIP in the message
                json_obj["message"] = json_obj["message"].replace(old, json_obj["serverIP"])
    if "streamName" in json_obj:
        if (not json_obj["streamName"]) or (json_obj["streamName"] == "unknown"):
            return False
        for stream in streamNames:
            if stream["streamName"] == json_obj["streamName"]:
                if stream["samples"] >= records_per_streamName:
                    return False
                stream["samples"] += 1
    else:
        return False
    # Replace the occurrence of any protected strings with a fake organization
    fake = random.choice(fake_orgs)
    for protected_string in protected_strings:
        json_obj["message"] = json_obj["message"].replace(protected_string, fake)
        upper_protected_string = protected_string.upper()
        json_obj["message"] = json_obj["message"].replace(upper_protected_string, fake.upper())

    #Unset the organization field
    if "organization" in json_obj:
        del json_obj["organization"]
    #Unset the pipelineEntrypoint
    if "pipelineEntrypoint" in json_obj:
        del json_obj["pipelineEntrypoint"]
    #Unset the pipelineName
    if "pipelineName" in json_obj:
        del json_obj["pipelineName"]
    #Unset the serverSetIds, clientSetIds, fileSetIds and userSetIds
    if "serverSetIds" in json_obj:
        del json_obj["serverSetIds"]
    if "clientSetIds" in json_obj:
        del json_obj["clientSetIds"]
    if "fileSetIds" in json_obj:
        del json_obj["fileSetIds"]
    if "userSetIds" in json_obj:
        del json_obj["userSetIds"]
    #Unset matchedLeadRuleIds and foreignIds
    if "matchedLeadRuleIds" in json_obj:
        del json_obj["matchedLeadRuleIds"]
    if "foreignIds" in json_obj:
        del json_obj["foreignIds"]
    #Unset the pipelineEntrypoint
    if "pipelineentrypoint" in json_obj:
        del json_obj["pipelineentrypoint"]
    #Unset the pipelineName
    if "pipelinename" in json_obj:
        del json_obj["pipelinename"]
    #Unset the serverSetIds, clientSetIds, fileSetIds and userSetIds
    if "serversetids" in json_obj:
        del json_obj["serversetids"]
    if "clientsetids" in json_obj:
        del json_obj["clientsetids"]
    if "filesetids" in json_obj:
        del json_obj["filesetids"]
    if "usersetids" in json_obj:
        del json_obj["usersetids"]
    #Unset matchedLeadRuleIds and foreignIds
    if "matchedleadruleids" in json_obj:
        del json_obj["matchedleadruleids"]
    if "foreignids" in json_obj:
        del json_obj["foreignids"]
    if "id_raw" in json_obj:
        del json_obj["id_raw"]
    if "partition" in json_obj:
        del json_obj["partition"]
    if "fieldextractorname" in json_obj:
        del json_obj["fieldextractorname"]
    # Convert the JSON object back to a string
    row = json.dumps(json_obj)
    # Return the sanitized row
    return row

def load_products():
    global products, streamNames
    # open each JSON file in the products directory
    for filename in os.listdir("products"):
        if filename.endswith(".json"):
            with open(os.path.join("products", filename), "r") as file:
                # load the JSON file
                product = json.load(file)
                # append the product to the list
                products.append(product)
                # if the list streamNames exists in the product, append the streamName to the list
                if "streamNames" in product:
                    for stream in product["streamNames"]:
                        streamDetails = {
                            "streamName": stream,
                            "product": product["name"],
                            "vendor": product["vendor_name"],
                            "samples": 0
                        }
                        streamNames.append(streamDetails)
    return products, streamNames

def streamName_to_product(streamName):
    global streamNames
    for stream in streamNames:
        if stream["streamName"] == streamName:
            return stream
    # Add a new product to the list
    new_product = {
        "streamName": streamName,
        "product": streamName,
        "vendor": streamName,
        "samples": 1
    }
    streamNames.append(new_product)

    return new_product

# Load the products and streamNames
products, streamNames = load_products()
def print_streamName_stats():
    global streamNames
    full = 0
    partital = 0
    total = 0
    for stream in streamNames:
        if stream["samples"] == 0:
            continue
        if stream["samples"] > 0 and stream["samples"] < records_per_streamName:
            partital += 1
        if stream["samples"] >= records_per_streamName:
            full += 1
        total += 1
    print(f"Total streamNames: {total}, Full: {full}, Partial: {partital}")
# Open CSV file
columns = ['instruction', 'input_text', 'output_text']
with open('output.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, escapechar='\\', quoting=csv.QUOTE_NONE)
    writer.writerow(columns)

    # Create an SSL context and skip certificate verification
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Initialize the cluster
    auth_provider = PlainTextAuthProvider(username=username, password=password)
    cluster = Cluster([cassandra_host], port=cassandra_port, auth_provider=auth_provider, ssl_context=ssl_context)

    try:
        session = cluster.connect('system')  # Replace 'your_keyspace' with your actual keyspace
        # Fetch and process rows
        created_rows = 0
        processed_artifacts = 0
        processed_incidents = 0
        # Execute a query to retrieve rows in batches of 10
        query = 'SELECT object FROM precinct.incidents'  # Replace 'your_table' with your actual table name
        statement = SimpleStatement(query, fetch_size=10)
        first_row = True
        for row in session.execute(statement):
            processed_incidents += 1
            # For every 1000 rows, print the number of rows processed
            if processed_incidents % 1000 == 0:
                formated_rows = "{:,}".format(created_rows*3)
                formated_artifacts = "{:,}".format(processed_artifacts)
                formated_incidents = "{:,}".format(processed_incidents)
                if processed_artifacts == 0:
                    skip_rate = 0
                else:
                    skip_rate = (processed_artifacts - created_rows) / processed_artifacts * 100
                # Get current date and time

                date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{date_time}: Created {formated_rows} rows from {formated_artifacts} artifacts and {formated_incidents} incidents. Skip rate: {skip_rate:.2f}%")
                print_streamName_stats()
            incident = json.loads(row.object)
            if "leads" not in incident:
                print("No artifacts found in the incident")
                continue
            leads = incident.get("leads")
            for item in leads:
                processed_artifacts += 1
                data = leads[item]['artifact']
                # Convert the data to a string removing escape characters
                data = json.dumps(data)
                data = sanitize(data)
                if not data:
                    continue
                # If streamName or message is not present, skip the row
                if "streamName" not in json.loads(data) or "message" not in json.loads(data):
                    print("StreamName or message not found in the artifact")
                    continue
                # Get the streamName from the row
                streamName = json.loads(data)["streamName"]
                message = json.loads(data)["message"]
                obj = json.loads(data)
                # foreach empty or null value, remove the key
                for key in list(obj.keys()):
                    if not obj[key]:
                        del obj[key]
                # Get the product details from the streamName
                product = streamName_to_product(streamName)
                # If the product is False, skip the row
                if not product:
                    print("Product not found for streamName: ", streamName)
                    continue
                product_name = product["product"]
                vendor_name = product["vendor"]

                # Build an instruction to create an artifact from the message
                instruction = "Create a JSON artifact from the message"
                input_data = message
                output_data = json.dumps(obj, indent=4)
                writer.writerow([instruction, input_data, output_data])

                # Build an instruction to identify the syslog message
                instruction = "Identify this syslog message"
                input_data = message
                output_data = f"Product: {product_name}\nVendor: {vendor_name}"
                writer.writerow([instruction, input_data, output_data])

                # Build an instruction to explain the message
                instruction = "Explain this syslog message"
                input_data = message
                output_data = f"This is a syslog message from a product called {product_name} by {vendor_name}. The following fields were extracted from the message:"
                for key in obj:
                    #Skip the message field
                    if key == "message":
                        continue
                    output_data += f"\n{key}: {obj[key]}"
                writer.writerow([instruction, input_data, output_data])
                created_rows += 1

                # If the number of created rows is greater than the max_records, break the loop
                if created_rows >= max_records:
                    break

        # Fetch artifacts from the database
        query = 'SELECT artifact_json FROM artifacts.artifacts'  # Replace 'your_table' with your actual table name
        statement = SimpleStatement(query, fetch_size=10)
        first_row = True
        for row in session.execute(statement):
            processed_artifacts += 1
            # For every 1000 rows, print the number of rows processed
            if processed_artifacts % 1000 == 0:
                formated_rows = "{:,}".format(created_rows*3)
                formated_artifacts = "{:,}".format(processed_artifacts)
                formated_incidents = "{:,}".format(processed_incidents)
                skip_rate = (processed_artifacts - created_rows) / processed_artifacts * 100
                # Get current date and time

                date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{date_time}: Created {formated_rows} rows from {formated_artifacts} artifacts and {formated_incidents} incidents. Skip rate: {skip_rate:.2f}%")
                print_streamName_stats()
            data = row.artifact_json
            # If streamName or message is not present, skip the row
            if "streamName" not in json.loads(data) or "message" not in json.loads(data):
                continue
            data = sanitize(data)
            if not data:
                continue
            # Get the streamName from the row
            streamName = json.loads(data)["streamName"]
            message = json.loads(data)["message"]
            obj = json.loads(data)
            # foreach empty or null value, remove the key
            for key in list(obj.keys()):
                if not obj[key]:
                    del obj[key]
            # Get the product details from the streamName
            product = streamName_to_product(streamName)
            # If the product is False, skip the row
            if not product:
                print("Product not found for streamName: ", streamName)
                continue
            product_name = product["product"]
            vendor_name = product["vendor"]

            # Build an instruction to create an artifact from the message
            instruction = "Create a JSON artifact from the message"
            input_data = message
            output_data = json.dumps(obj, indent=4)
            writer.writerow([instruction, input_data, output_data])

            # Build an instruction to identify the syslog message
            instruction = "Identify this syslog message"
            input_data = message
            output_data = f"Product: {product_name}\nVendor: {vendor_name}"
            writer.writerow([instruction, input_data, output_data])

            # Build an instruction to explain the message
            instruction = "Explain this syslog message"
            input_data = message
            output_data = f"This is a syslog message from a product called {product_name} by {vendor_name}. The following fields were extracted from the message:"
            for key in obj:
                #Skip the message field
                if key == "message":
                    continue
                output_data += f"\n{key}: {obj[key]}"
            writer.writerow([instruction, input_data, output_data])
            created_rows += 1
            # If the number of created rows is greater than the max_records, break the loop
            if created_rows >= max_records:
                break
        print("Data processing complete")
        formated_rows = "{:,}".format(created_rows*3)
        formated_artifacts = "{:,}".format(processed_artifacts)
        skip_rate = (processed_artifacts - created_rows) / processed_artifacts * 100
        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{date_time}: Created {formated_rows} rows from {formated_artifacts} artifacts. Skip rate: {skip_rate:.2f}%")
        print_streamName_stats()
        run_time = datetime.datetime.now() - start_time
        #format the run time in hours, minutes and seconds
        run_time = str(run_time).split(".")[0]
        print(f"Total run time: {run_time}")
        #Save the streamNames to a file
        with open('streamNames.json', 'w') as outfile:
            json.dump(streamNames, outfile)
    finally:
        print("Closing the connection")
        cluster.shutdown()