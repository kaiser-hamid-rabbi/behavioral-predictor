"""Generate static features for all users and save to Feature Store."""

import asyncio

from app.db.base import async_session_factory, dispose_engine
from app.db.repositories.user_repository import UserRepository
from app.db.repositories.event_repository import EventRepository
from app.db.repositories.feature_repository import FeatureRepository
from app.ml.feature_engineering.feature_builder import FeatureBuilder

async def main():
    print("Generating user features...")
    
    async with async_session_factory() as session:
        user_repo = UserRepository(session)
        event_repo = EventRepository(session)
        feature_repo = FeatureRepository(session)
        
        users = await user_repo.get_all(limit=1000)
        
        for user in users:
            events = await event_repo.get_by_user(user.id, limit=2000)
            
            # Convert ORM events to dicts/pandas
            import pandas as pd
            if not events:
                continue
                
            df = pd.DataFrame([{
                'event_time': e.event_time,
                'event_name': e.event_name,
                'category': e.category,
                'session_id': e.session_id,
                'channel': e.channel,
                'device_os': e.device_os,
            } for e in events])
            
            features = FeatureBuilder.build_user_features(df)
            
            import datetime
            await feature_repo.upsert(
                user.id,
                features,
                version=1,
                computed_at=datetime.datetime.now(datetime.timezone.utc)
            )
            
        await session.commit()
        print(f"Generated features for {len(users)} users")
        
    await dispose_engine()

if __name__ == "__main__":
    asyncio.run(main())
