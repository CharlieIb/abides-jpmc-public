import pandas as pd
import argparse
import sys
import traceback


def analyze_exchange_log(exchange_log_path: str, summary_log_path: str):
    """
    Loads and analyzes the ExchangeAgent log, using the summary_log to map
    agent IDs to their strategy names.

    Args:
        exchange_log_path (str): Path to the EXCHANGE_AGENT.bz2 log file.
        summary_log_path (str): Path to the summary_log.bz2 file.
    """
    try:
        # --- Step 1: Build the Agent ID -> Strategy Name Map ---
        print(f"--- Loading summary log from: {summary_log_path} ---")
        summary_df = pd.read_pickle(summary_log_path, compression='bz2')

        # Filter for the 'STARTING_CASH' event, which has the info we need
        agent_info_df = summary_df[summary_df['EventType'] == 'STARTING_CASH']

        # Create the dictionary mapping
        id_to_strategy_map = pd.Series(
            agent_info_df.AgentStrategy.values,
            index=agent_info_df.AgentID
        ).to_dict()
        print("Agent ID to Strategy map created successfully.")

        # --- Step 2: Load and Parse the Exchange Log ---
        print(f"\n--- Loading Exchange log from: {exchange_log_path} ---")
        exchange_df = pd.read_pickle(exchange_log_path, compression='bz2')
        print("Exchange log loaded successfully.")

        print("\n--- Parsing event data from exchange log... ---")
        parsed_data = []
        for timestamp, row in exchange_df.iterrows():
            event_type = row['EventType']
            event_dict = row['Event']

            # We only care about executed orders for this analysis
            if event_type == 'OrderExecutedMsg':
                # Create a record using dictionary key access
                record = {
                    'Timestamp': timestamp,
                    'AgentID': event_dict.get('agent_id'),
                    'OrderID': event_dict.get('order_id'),
                    'Price': event_dict.get('fill_price'),
                    'Quantity': event_dict.get('quantity'),
                    # The 'side' is an enum, so we access its .name property
                    'Side': event_dict.get('side').name if hasattr(event_dict.get('side'), 'name') else None
                }
                parsed_data.append(record)

        if not parsed_data:
            print("--- No 'OrderExecutedMsg' events found in the log. ---")
            return True

        # Create a clean DataFrame of all executed trades
        fills_df = pd.DataFrame(parsed_data)
        print(f"Parsed {len(fills_df)} executed trades.")

        # --- Step 3: Enrich and Analyze the Trade Data ---
        fills_df['Timestamp'] = pd.to_datetime(fills_df['Timestamp'])

        # Use the map from Step 1 to add the 'AgentStrategy' column
        fills_df['AgentStrategy'] = fills_df['AgentID'].map(id_to_strategy_map)

        print("\n\n--- Analysis of Executed Trades ---")

        print("\n--- Total Executed Trades by Agent Strategy ---")
        print(fills_df['AgentStrategy'].value_counts())

        # print("\n--- Executed Trades for FinancialGymAgent ---")
        # gym_agent_trades = fills_df[fills_df['AgentStrategy'] == 'FinancialGymAgent']
        #
        # if not gym_agent_trades.empty:
        #     print(f"Found {len(gym_agent_trades)} executed trades for the Gym Agent:")
        #     display_cols = ['Timestamp', 'Side', 'Quantity', 'Price']
        #     print(gym_agent_trades[display_cols].to_string())
        # else:
        #     print("No executed trades found for FinancialGymAgent.")

    except FileNotFoundError as e:
        print(f"Error: A log file was not found.", file=sys.stderr)
        print(e, file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read and analyze ABIDES ExchangeAgent and summary log files.")
    parser.add_argument("exchange_log", type=str, help="Path to the EXCHANGE_AGENT.bz2 log file.")
    parser.add_argument("summary_log", type=str, help="Path to the summary_log.bz2 file.")
    args = parser.parse_args()

    if not analyze_exchange_log(args.exchange_log, args.summary_log):
        sys.exit(1)
