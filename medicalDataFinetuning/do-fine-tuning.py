import jsonlines
import os
import sys
import time
import csv
import json
import openai
from dotenv import load_dotenv
load_dotenv()

model = "gpt-3.5-turbo"  # Change this to the model you want to fine-tune

def setup_api_key():
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("API key set up successfully.")

def create_fine_tuning_file(file_path):
    print("Processing fine tuning file " + file_path)
    file = openai.File.create(
        file=open(file_path, "rb"),
        purpose='fine-tune'
    )

    # Get the file ID
    file_id = file['id']

    # Check the file's status
    status = file['status']

    while status != 'processed':
        print(f"File status: {status}. Waiting for the file to be processed...")
        time.sleep(10)  # Wait for 10 seconds
        file_response = openai.File.retrieve(file_id)
        status = file_response['status']
        print(file_response)
    fine_tuning_response = openai.FineTuningJob.create(training_file=file_id, model=model)
    print(fine_tuning_response)
    return fine_tuning_response

def fine_tune_model(fine_tuning_file):
    print("Starting fine tuning job with ID: " + fine_tuning_file['id'])
    if fine_tuning_file['status'] == 'processed':
        fine_tuning_response = openai.FineTuningJob.create(
            training_file=fine_tuning_file['id'],
            model=model
        )
        print(fine_tuning_response['id'])

def load_csv_finetuning(csv_file, output_path):
    print("Loading CSV fine tuning data from " + csv_file)
    # Open the CSV file for reading
    with open(csv_file, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Open the JSONL file for writing
        with jsonlines.open(output_path, mode='w') as jsonl_file:
            for row in csv_reader:
                system = row[0]
                values = [{"role": "system", "content": system}]
                odd = True
                for value in row[1:]:
                    if odd:
                        if len(value) > 0:
                            values.append({"role": "user", "content": value})
                        odd = False
                    else:
                        if len(value) > 0:
                            values.append({"role": "assistant", "content": value})
                        odd = True
                json_data = {"messages":values}
                jsonl_file.write(json_data)

if __name__ == '__main__':
    setup_api_key()
    
    fine_tuning_data = "./medicalDataFinetuning/output/train.jsonl"
    if not os.path.exists(fine_tuning_data):
        print("Fine tuning data file does not exist. Creating it.")
        # Create the fine-tuning data file
        with open(fine_tuning_data, 'w') as f:
            pass
    csv_file = "./medicalDataFinetuning/data/train.csv"

    load_csv_finetuning(csv_file, fine_tuning_data)
   
    fine_tuning_file = create_fine_tuning_file(fine_tuning_data)
    print('fine_tuning_file:', fine_tuning_file)
    #save fine_tuning_file to json file with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(f"medicalDataFinetuning/output/finetuningFiles/fine_tuning_file_{timestamp}.json", "w") as f:
        json.dump(fine_tuning_file, f, indent=4)
        
    id = fine_tune_model(fine_tuning_file)
    print(openai.FineTuningJob.list(limit=10))