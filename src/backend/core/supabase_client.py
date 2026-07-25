from supabase import create_client, Client

from src.backend.core.config import SUPABASE_URL, SUPABASE_KEY

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL / SUPABASE_KEY are not set. Copy .env.example to .env "
        "and fill in your Supabase project's URL and API key."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
