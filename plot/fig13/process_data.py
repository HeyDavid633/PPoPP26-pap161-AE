import csv
import re
import ast

def read_tuning_cost(file_path):
    tuning_data = {}
    
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                input_size = ast.literal_eval(row['Input Size'])
                batch_size, seq_length = input_size
                
                model_name = row['Model Name']
                stof_value = float(row['STOF(ours)'])
                key = (batch_size, seq_length, model_name)
                tuning_data[key] = stof_value * 1000
       
                
    except FileNotFoundError:
        print(f"Error: Cannot find file {file_path}")
    except Exception as e:
        print(f"Error reading Tuning_cost.csv: {e}")
    
    return tuning_data

def process_overhead_data_with_tuning_cost(tuning_data):
    model_mapping_to_tuning = {
        'bert_base': 'BERT-Base',
        'bert_large': 'BERT-Large', 
        'gpt': 'GPT',
        'llama_base': 'LLaMA',
        't5': 'T5',
        'vit_base': 'ViT'
    }
    
    model_mapping_to_output = {
        'BERT-Base': 'Bert-base',
        'BERT-Large': 'Bert-large', 
        'GPT': 'GPT-2',
        'LLaMA': 'LLaMA',
        'T5': 'T5',
        'ViT': 'ViT'
    }
    
    try:
        with open('../../data/Overhead_Analysis/overhead_analysis_raw.csv', 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("error: Cant find '../../data/Overhead_Analysis/overhead_analysis_raw.csv'")
        return
    
    processed_data = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        pattern = r'(.+?)\s*\|\s*bs:(\d+)\s*\|\s*seq:(\d+)\s*\|\s*analysis_model\s*([\d.]+)\s*ms\s*\|\s*encoding:\s*([\d.]+)\s*ms\s*\|\s*decoding:\s*([\d.]+)\s*ms\s*\|\s*reward:\s*([\d.]+)\s*ms'
        match = re.match(pattern, line)
        
        if match:
            model_raw, batch_size, seq_length, analysis_model, encoding, decoding, reward = match.groups()
            model_for_lookup = model_mapping_to_tuning.get(model_raw, model_raw)
            
            batch_size = int(batch_size)
            seq_length = int(seq_length)
            analysis_model = float(analysis_model)
            encoding = float(encoding)
            decoding = float(decoding)
            reward = float(reward)

            key = (batch_size, seq_length, model_for_lookup)
            if key in tuning_data:
                stof = tuning_data[key]
                print(f"Using Tuning Cost STOF for {key}: {stof}")

                output_model = model_mapping_to_output.get(model_for_lookup, model_for_lookup)
                
                processed_data.append([
                    batch_size, seq_length, output_model, 
                    round(analysis_model, 3), 
                    round(encoding, 3), 
                    round(decoding, 3), 
                    round(reward, 3), 
                    round(stof, 2)
                ])
            else:
     
                print(f"Error: No Tuning Cost data found for {key}. Skipping this entry.")
                continue

    with open('../../data/Overhead_Analysis/Overhead_analysis.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Batch Size', 'Sequence Length', 'Model', 
            'Analytical Model', 'Hash Encoding', 'Numercial Decoding',
            'Reward Algorithm', 'STOF'
        ])
        writer.writerows(processed_data)

if __name__ == "__main__":
    tuning_file_path = "../../data/Tuning_Cost/tuning_cost.csv"
    tuning_data = read_tuning_cost(tuning_file_path)

    process_overhead_data_with_tuning_cost(tuning_data)
    
    print("\n=== Processing Complete ===")
    print("Overhead_analysis.csv has been created with STOF values from tuning_cost.csv")