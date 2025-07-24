import pandas as pd
import argparse
import sys
import traceback


def analyze_exchange_log(exchange_log_path: str, summary_log_path: str):
    """
    Loads and analyzes simulation logs to calculate agent performance and profit.

    Args:
        exchange_log_path (str): Path to the EXCHANGE_AGENT.bz2 log file.
        summary_log_path (str): Path to the summary_log.bz2 file.
    """
    try:
        # --- Step 1: Load both log files ---
        print(f"--- Loading summary log from: {summary_log_path} ---")
        summary_df = pd.read_pickle(summary_log_path, compression='bz2')

        print(f"--- Loading Exchange log from: {exchange_log_path} ---")
        exchange_df = pd.read_pickle(exchange_log_path, compression='bz2')

        # --- Step 2: Build the Agent ID -> Strategy Name Map ---
        agent_info_df = summary_df[summary_df['EventType'] == 'STARTING_CASH']
        id_to_strategy_map = pd.Series(
            agent_info_df.AgentStrategy.values,
            index=agent_info_df.AgentID
        ).to_dict()
        print("\n--- Agent ID to Strategy map created successfully. ---")

        # --- Step 3: Parse the Exchange Log for Executed Trades ---
        print("--- Parsing event data from exchange log... ---")
        parsed_data = []
        for timestamp, row in exchange_df.iterrows():
            if row['EventType'] == 'OrderExecutedMsg':
                event_dict = row['Event']
                record = {
                    'Timestamp': timestamp,
                    'AgentID': event_dict.get('agent_id'),
                    'Price': event_dict.get('fill_price'),
                    'Quantity': event_dict.get('quantity'),
                    'Side': event_dict.get('side').name if hasattr(event_dict.get('side'), 'name') else None
                }
                parsed_data.append(record)

        if not parsed_data:
            print("--- No 'OrderExecutedMsg' events found in the log. ---")
            return True

        fills_df = pd.DataFrame(parsed_data)
        fills_df['Timestamp'] = pd.to_datetime(fills_df['Timestamp'])
        fills_df['AgentStrategy'] = fills_df['AgentID'].map(id_to_strategy_map)
        print(f"Parsed {len(fills_df)} executed trades.")

        # --- Step 4: Calculate Profit for the FinancialGymAgent ---
        print("\n\n--- Profit Calculation for FinancialGymAgent ---")

        gym_agent_trades = fills_df[fills_df['AgentStrategy'] == 'FinancialGymAgent'].copy()

        if gym_agent_trades.empty:
            print("No executed trades found for FinancialGymAgent. Cannot calculate profit.")
        else:
            # Calculate Realized Profit from trades
            gym_agent_trades['CostBasis'] = gym_agent_trades['Price'] * gym_agent_trades['Quantity']

            total_buys = gym_agent_trades[gym_agent_trades['Side'] == 'BID']['CostBasis'].sum()
            total_sells = gym_agent_trades[gym_agent_trades['Side'] == 'ASK']['CostBasis'].sum()
            realized_profit = total_sells - total_buys

            print(f"Total Buy Cost:  ${total_buys:,.2f}")
            print(f"Total Sell Revenue: ${total_sells:,.2f}")
            print(f"Realized Profit (from trades): ${realized_profit:,.2f}")

        # Calculate Total PnL from summary log
        gym_agent_summary = summary_df[summary_df['AgentStrategy'] == 'FinancialGymAgent']

        try:
            start_event = gym_agent_summary[gym_agent_summary['EventType'] == 'STARTING_CASH']['Event'].iloc[0]
            final_event = gym_agent_summary[gym_agent_summary['EventType'] == 'FINAL_STATE']['Event'].iloc[0]

            starting_cash = start_event
            final_portfolio_value = final_event['final_portfolio_value']
            total_pnl = final_portfolio_value - starting_cash

            print("\n--- Total PnL Calculation (from summary_log) ---")
            print(f"Starting Portfolio Value: ${starting_cash:,.2f}")
            print(f"Final Portfolio Value:    ${final_portfolio_value:,.2f}")
            print(f"Total PnL:                ${total_pnl:,.2f}")

        except IndexError:
            print("\nCould not find STARTING_CASH or FINAL_STATE events in summary log.")
            print("Cannot calculate Total PnL.")

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
    parser = argparse.ArgumentParser(description="Read and analyze ABIDES logs to calculate agent profit.")
    parser.add_argument("exchange_log", type=str, help="Path to the EXCHANGE_AGENT.bz2 log file.")
    parser.add_argument("summary_log", type=str, help="Path to the summary_log.bz2 file.")
    args = parser.parse_args()

    if not analyze_exchange_log(args.exchange_log, args.summary_log):
        sys.exit(1)
