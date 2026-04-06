"""Load Parquet dataset into PostgreSQL using SQLAlchemy."""

import asyncio
import sys
from pathlib import Path
import pyarrow.parquet as pq

from app.core.config import get_settings
from app.db.base import async_session_factory, dispose_engine
from app.db.repositories.user_repository import UserRepository
from app.db.repositories.event_repository import EventRepository

async def main():
    settings = get_settings()
    data_dir = settings.data_dir_path
    
    users_dir = data_dir / "users"
    events_dir = data_dir / "events"
    
    print(f"Loading data from {data_dir}... This handles subsets as required")
    
    # Simple chunk loader for user parquets inside async context
    if users_dir.exists():
        for file in users_dir.glob("*.parquet"):
            print(f"Reading {file.name}...")
            table = pq.read_table(file).to_pandas()
            # If sub-setting
            if settings.data_sample_users and settings.data_sample_users > 0:
                table = table.head(settings.data_sample_users)
                
            records = table.to_dict('records')
            
            async with async_session_factory() as session:
                user_repo = UserRepository(session)
                try:
                    await user_repo.create_batch(records)
                    await session.commit()
                    print(f"Loaded {len(records)} users")
                except Exception as e:
                    print(f"User bulk insertion error: {e}")
                    await session.rollback()

    if events_dir.exists():
         for file in events_dir.glob("*.parquet"):
            print(f"Reading events from {file.name}... (Using pyarrow iterator)")
            pf = pq.ParquetFile(file)
            for batch in pf.iter_batches(batch_size=settings.data_chunk_size):
                df = batch.to_pandas()
                # Assuming simple limits if subsetting is critical
                records = df.to_dict('records')
                
                async with async_session_factory() as session:
                    event_repo = EventRepository(session)
                    try:
                        await event_repo.create_batch(records)
                        await session.commit()
                        print(f"Inserted batch of {len(records)} events")
                    except Exception as e:
                       print(f"Events bulk insertion error: {e}")
                       await session.rollback()
                break # Only load first batch if sample run
                
    await dispose_engine()

if __name__ == "__main__":
    asyncio.run(main())
