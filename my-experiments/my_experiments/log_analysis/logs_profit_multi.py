import pandas as pd
import argparse
import sys
import traceback


def calculate_and_print_pnl(agent_name: str, all_fills_df: pd.DataFrame, all_summary_df: pd.DataFrame):
    """
    Filters logs for a specific agent, calculates its profit, and prints the results.

    Args:
        agent_name (str): The 'AgentStrategy' name to analyze.
        all_fills_df (pd.DataFrame): DataFrame containing all executed trades.
        all_summary_df (pd.DataFrame): DataFrame containing the full summary log.
    """
    print(f"\n\n--- Profit Calculation for {agent_name} ---")

    # --- 1. Calculate Realized Profit from Trades ---
    agent_trades = all_fills_df[all_fills_df['AgentStrategy'] == agent_name].copy()

    if agent_trades.empty:
        print(f"No executed trades found for {agent_name}. Cannot calculate realized profit.")
    else:
        agent_trades['CostBasis'] = agent_trades['Price'] * agent_trades['Quantity']

        total_buys = agent_trades[agent_trades['Side'] == 'BID']['CostBasis'].sum()/100
        total_sells = agent_trades[agent_trades['Side'] == 'ASK']['CostBasis'].sum()/100
        realized_profit = total_sells - total_buys

        print("\n--- Realized Profit (from trades) ---")
        print(f"Total Buy Cost:     ${total_buys:,.2f}")
        print(f"Total Sell Revenue: ${total_sells:,.2f}")
        print(f"Realized Profit:    ${realized_profit:,.2f}")

    # --- 2. Calculate Total PnL from Summary Log ---
    agent_summary = all_summary_df[all_summary_df['AgentStrategy'] == agent_name]

    try:
        start_event = agent_summary[agent_summary['EventType'] == 'STARTING_CASH']['Event'].iloc[0]
        final_event = agent_summary[agent_summary['EventType'] == 'ENDING_CASH']['Event'].iloc[0]

        # Convert from cents to dollars for calculation and display
        starting_cash = start_event / 100
        final_portfolio_value = final_event / 100
        total_pnl = final_portfolio_value - starting_cash

        print("\n--- Total PnL (from summary_log) ---")
        print(f"Starting Portfolio Value: ${starting_cash:,.2f}")
        print(f"Final Portfolio Value:    ${final_portfolio_value:,.2f}")
        print(f"Total PnL:                ${total_pnl:,.2f}")

    except IndexError:
        print("\nCould not find STARTING_CASH or ENDING_CASH events in summary log.")
        print("Cannot calculate Total PnL.")


def analyze_simulation_logs(exchange_log_path: str, summary_log_path: str):
    """
    Main function to load and analyze simulation logs.
    """
    try:
        # --- Load Logs and Create Agent Map ---
        print(f"--- Loading summary log from: {summary_log_path} ---")
        summary_df = pd.read_pickle(summary_log_path, compression='bz2')

        print(f"--- Loading Exchange log from: {exchange_log_path} ---")
        exchange_df = pd.read_pickle(exchange_log_path, compression='bz2')

        agent_info_df = summary_df[summary_df['EventType'] == 'STARTING_CASH']
        id_to_strategy_map = pd.Series(
            agent_info_df.AgentStrategy.values,
            index=agent_info_df.AgentID
        ).to_dict()
        print("\n--- Agent ID to Strategy map created successfully. ---")

        # --- Parse Exchange Log for Executed Trades ---
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
        fills_df['AgentStrategy'] = fills_df['AgentID'].map(id_to_strategy_map)
        print(f"Parsed {len(fills_df)} executed trades.")
        print("\n\n--- Overall Market Statistics ---")
        if not fills_df.empty:
            # Note: Prices in ABIDES are in cents. Divide by 100 for dollar values.
            price_series_dollars = fills_df['Price'] / 100

            mean_price = price_series_dollars.mean()
            std_dev_price = price_series_dollars.std()

            print(f"Mean Execution Price:      ${mean_price:,.2f}")
            print(f"Standard Deviation of Price: ${std_dev_price:,.2f}")
        else:
            print("No trades were executed, cannot calculate market statistics.")
        # --- Analyze Each Agent of Interest ---
        agents_to_analyze = [
            'FinancialGymAgent',
            'ValueAgent',
            'MomentumAgent',
            'AdaptivePOVMarketMakerAgent'
        ]

        for agent_name in agents_to_analyze:
            calculate_and_print_pnl(agent_name, fills_df, summary_df)

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

    if not analyze_simulation_logs(args.exchange_log, args.summary_log):
        sys.exit(1)
