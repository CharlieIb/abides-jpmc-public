import pandas as pd
import argparse
import sys
import traceback


def analyze_exchange_log(log_file_path: str):
    """
    Diagnoses the structure of an ABIDES ExchangeAgent log file by inspecting
    the 'Event' objects.
    """
    try:
        print(f"--- Loading Exchange log from: {log_file_path} ---")
        df = pd.read_pickle(log_file_path, compression='bz2')
        print("Log file loaded successfully.")

        # --- Diagnostic Step: Inspect the Event objects ---
        print("\n--- Starting Diagnostic Inspection ---")

        # We will limit how many events we inspect to keep the output clean.
        events_to_inspect = {
            'OrderAcceptedMsg': 2,
            'OrderExecutedMsg': 2,
            'LimitOrderMsg': 2,
        }

        inspected_counts = {k: 0 for k in events_to_inspect}

        for index, row in df.iterrows():
            event_type = row['EventType']

            if event_type in events_to_inspect and inspected_counts[event_type] < events_to_inspect[event_type]:
                event = row['Event']

                print(f"\n--- Inspecting an '{event_type}' event ---")

                # 1. Print the type of the 'Event' object
                print(f"  - Type: {type(event)}")

                # 2. Print the object itself to see its string representation
                print(f"  - Representation: {event}")

                # 3. Print all available attributes and methods of the object
                print(f"  - Attributes (dir()): {dir(event)}")

                # If the event is an object with attributes, let's try to access a common one.
                if hasattr(event, 'order'):
                    print("  - It has an 'order' attribute. Inspecting dir(event.order):")
                    print(f"    - {dir(event.order)}")

                inspected_counts[event_type] += 1

        print("\n--- Diagnostic Inspection Complete ---")

        # Check if all requested events were found and inspected
        for event_type, count in inspected_counts.items():
            if count == 0:
                print(f"Warning: Did not find any '{event_type}' events to inspect.")

    except FileNotFoundError:
        print(f"Error: The file was not found at the specified path: {log_file_path}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diagnose ABIDES ExchangeAgent log files.")
    parser.add_argument("log_file", type=str, help="Path to the EXCHANGE_AGENT.bz2 log file.")
    args = parser.parse_args()

    if not analyze_exchange_log(args.log_file):
        sys.exit(1)
