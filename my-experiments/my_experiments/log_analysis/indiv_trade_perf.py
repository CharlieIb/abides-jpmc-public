import pandas as pd
import argparse
import sys
import traceback

def analyze_agent_trades(agent_name: str, agent_fills_df: pd.DataFrame):
    """
    Analyzes the individual trades of a specific agent and prints performance statistics.

    Args:
        agent_name (str): The 'AgentStrategy' name to analyze.
        agent_fills_df (pd.DataFrame): DataFrame containing only the trades for this agent.
    """
    print(f"\n\n--- Trade Performance Analysis for: {agent_name} ---")

    if agent_fills_df.empty:
        print("No trades found for this agent.")
        return

    # --- Calculate Profit/Loss for each individual trade ---
    # A positive value is a profit (sell), a negative value is a cost (buy).
    agent_fills_df['TradeValue'] = agent_fills_df.apply(
        lambda row: (row['Price'] / 100) * row['Quantity'] if row['Side'] == 'ASK' else -((row['Price'] / 100) * row['Quantity']),
        axis=1
    )

    # Separate into winning (profitable sells) and losing (costly buys) trades
    # For this analysis, we'll consider any sell a "win" and any buy a "loss" in terms of cash flow.
    winning_trades = agent_fills_df[agent_fills_df['TradeValue'] > 0]
    losing_trades = agent_fills_df[agent_fills_df['TradeValue'] < 0]

    num_trades = len(agent_fills_df)
    num_wins = len(winning_trades)
    num_losses = len(losing_trades)

    print(f"Total Trades: {num_trades}")
    print(f"  - Winning Trades (Sells): {num_wins}")
    print(f"  - Losing Trades (Buys):   {num_losses}")

    if num_wins > 0:
        largest_profit = winning_trades['TradeValue'].max()
        avg_win = winning_trades['TradeValue'].mean()
        print(f"Largest Single Profit (Sell): ${largest_profit:,.2f}")
        print(f"Average Profit per Winning Trade: ${avg_win:,.2f}")

    if num_losses > 0:
        largest_loss = losing_trades['TradeValue'].min()
        avg_loss = losing_trades['TradeValue'].mean()
        print(f"Largest Single Loss (Buy Cost): ${largest_loss:,.2f}")
        print(f"Average Loss per Losing Trade:  ${avg_loss:,.2f}")


def analyze_simulation_logs(exchange_log_path: str, summary_log_path: str):
    """
    Main function to load logs and orchestrate the analysis.
    """
    try:
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

        # --- Analyze Each Agent of Interest ---
        agents_to_analyze = fills_df['AgentStrategy'].unique()
        print(f"\nFound agent strategies to analyze: {list(agents_to_analyze)}")

        for agent_name in agents_to_analyze:
            agent_specific_fills = fills_df[fills_df['AgentStrategy'] == agent_name]
            analyze_agent_trades(agent_name, agent_specific_fills)

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
    parser = argparse.ArgumentParser(description="Analyze individual trade performance from ABIDES logs.")
    parser.add_argument("exchange_log", type=str, help="Path to the EXCHANGE_AGENT.bz2 log file.")
    parser.add_argument("summary_log", type=str, help="Path to the summary_log.bz2 file.")
    args = parser.parse_args()

    if not analyze_simulation_logs(args.exchange_log, args.summary_log):
        sys.exit(1)
