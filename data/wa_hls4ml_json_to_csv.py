# data = categorized_models["Quarks"]

import json
import csv
import numpy as np
import re
import os 

def data_reader(file):
    with open(file, 'r') as json_file:
        data = json.load(json_file)
    return data

def produce_model_string(json):

    layers = []
    first_layer = None
    last_layer = None

    for layer in json:
        if layer['class_name'] != 'QDense':
            continue

        if first_layer is None:
            first_layer = layer['input_shape'][1]

        layers.append(str(layer['input_shape'][1]))
        last_layer = layer['output_shape'][1]

    layers.append(str(last_layer))

    assert(first_layer != None and last_layer != None)

    return "-".join(layers), first_layer, last_layer


def parse_file(input_path, output_csv="auto_parsed_json.csv"):
    
    csv_header = [
        'model_name', 'd_in', 'd_out', 'prec', 'model_file', 'model_string', 'rf', 'strategy', 
        'TargetClockPeriod_hls', 'EstimatedClockPeriod_hls', 
        'BestLatency_hls', 'WorstLatency_hls', 'IntervalMin_hls', 'IntervalMax_hls', 
        'BRAM_18K_hls', 'DSP_hls', 'FF_hls', 'LUT_hls', 'URAM_hls', "rf_times_precision", "hls_synth_success"]

    # Identify files first
    files_to_process = []
    if os.path.isdir(input_path):
        print(f"Scanning directory: {input_path}")
        for filename in os.listdir(input_path):
            if filename.endswith(".json"):
                 files_to_process.append(os.path.join(input_path, filename))
    else:
        files_to_process.append(input_path)

    if not files_to_process:
        print("No JSON files found to process.")
        return

    # Helper function to process a single list of data points and write to writer
    def process_data_list(data_list, writer, precision_example_ref):
         count = 0
         # Try to determine precision from first item if we haven't yet
         if precision_example_ref[0] is None and len(data_list) > 0:
            try:
                prec_dict = data_list[0]['hls_config'].get("LayerName", {})
                for key in prec_dict:
                     if "Precision" in prec_dict[key] and 'weight' in prec_dict[key]["Precision"]:
                         precision_example_ref[0] = prec_dict[key]["Precision"]['weight']
                         break
            except:
                pass
            if precision_example_ref[0] is None:
                precision_example_ref[0] = "ap_fixed<16,6>" # fallback

         for data_point in data_list:
            try:
                meta_data = data_point['meta_data']
                model_name = meta_data['model_name']
                model_config = data_point['model_config']
                
                model_string, d_in, d_out = produce_model_string(model_config)
                hls_config = data_point['hls_config']

                # Precision extreaction logic
                prec_val = precision_example_ref[0]
                try:
                    prec_dict = hls_config.get("LayerName", {})
                    for key in prec_dict:
                         if "Precision" in prec_dict[key] and 'weight' in prec_dict[key]["Precision"]:
                             prec_val = prec_dict[key]["Precision"]['weight']
                             break
                except:
                     pass
                
                prec_cleaned = re.sub(r",[0-9]+\>", "", prec_val)
                prec_cleaned = re.sub(r"fixed\<", "", prec_cleaned)
                curr_prec = "16" 
                
                l_names = hls_config.get("LayerName", {})
                keys = list(l_names.keys())
                # logic to try and get specific precision if multiple
                if len(keys) > 1:
                     target_key = keys[1]
                     if 'Precision' in l_names[target_key] and 'weight' in l_names[target_key]['Precision']:
                         curr_prec = l_names[target_key]['Precision']['weight']
                elif len(keys) == 1:
                     target_key = keys[0]
                     if 'Precision' in l_names[target_key] and 'weight' in l_names[target_key]['Precision']:
                         curr_prec = l_names[target_key]['Precision']['weight']

                curr_prec = re.sub(r",[0-9]+\>", "", curr_prec)
                curr_prec = re.sub(r"fixed\<", "", curr_prec)
                curr_prec_digits = re.sub(r"[^0-9]", "", curr_prec)
                if not curr_prec_digits: curr_prec_digits = "16"

                model_file = meta_data['artifacts_file']
                rf = hls_config['Model'].get('ReuseFactor', 1)
                strategy = hls_config['Model'].get('Strategy', "")

                latency_report = data_point['latency_report']
                target_clock = latency_report.get('target_clock', 0)
                estimated_clock = latency_report.get('estimated_clock', 0)
                best_latency = latency_report.get('cycles_min', 0)
                worst_latency = latency_report.get('cycles_max', 0)

                resource_report = data_point['resource_report']
                bram = resource_report.get('BRAM', resource_report.get('bram', 0))
                dsp = resource_report.get('DSP', resource_report.get('dsp', 0))
                ff = resource_report.get('FF', resource_report.get('ff', 0))
                lut = resource_report.get('LUT', resource_report.get('lut', 0))
                uram = resource_report.get('URAM', resource_report.get('uram', 0))

                rf_times_precision = int(curr_prec_digits) * int(rf)
                hls_synth_success = "TRUE"

                csv_row = [
                model_name, d_in, d_out, curr_prec_digits, model_file, model_string, rf, strategy,
                target_clock, estimated_clock, best_latency, worst_latency, best_latency, worst_latency,
                bram, dsp, ff, lut, uram, rf_times_precision, hls_synth_success]

                writer.writerow(csv_row)
                count += 1
            except Exception as e:
                pass
         return count


    total_models = 0
    precision_ref = [None] # Mutable ref to share across files if needed

    print(f"Writing to output CSV: {output_csv}")
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

        for filepath in files_to_process:
            print(f"Processing file: {filepath}")
            try:
                # Load one file into memory
                file_data = data_reader(filepath)
                # Process and write
                num_written = process_data_list(file_data, writer, precision_ref)
                total_models += num_written
                print(f"  - Extracted {num_written} models.")
                
                # Help GC
                del file_data
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    print(f'Parsing successful. Processed {len(files_to_process)} files. Total models: {total_models}. Output: "{output_csv}"')