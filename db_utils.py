import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    # Connect to the MySQL database using environment variables
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 25060))
    )
    return conn

def query_previous_strategies(session_id: str, country_code: str):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = """
    SELECT s.strategy_name, s.strategy_description, so.gdp_change, so.political_boost_loss, so.trade_balance_shift, so.total_payoff
    FROM strategies s
    LEFT JOIN strategy_outcomes so ON s.strategy_id = so.strategy_id
    WHERE s.session_id = %s
    AND s.country_code = %s
    AND s.is_active = TRUE
    ORDER BY s.created_at DESC;
    """
    cursor.execute(query, (session_id, country_code))
    strategies = cursor.fetchall()
    conn.close()
    return strategies 