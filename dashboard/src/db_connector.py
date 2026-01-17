import os 
from dotenv import load_dotenv
from supabase import create_client, Client 

load_dotenv() 

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key: 
    raise ValueError("Superbase Credentials missing in .env")
supabase: Client = create_client(url, key)

def fetch_user_history(user_id: str): 
    """
    Connects to Supabase and fetches history 
    """

    try: 
        response = supabase.table("workout_sessions") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(20) \
            .execute()
        data = response.data

        if not data: 
            print(f"No history found for user {user_id}")
            return []

        print(f"Successfully fetched {len(data)} sessions from Supabase. ")
        return data

    except Exception as e: 
        print (f"Database Error: {e}")
        return []
    
    