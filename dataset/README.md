# Dataset Directory

This directory is where the system expects to find the raw `.parquet` behavioral data.

Due to GitHub file size limits, the large parquet files are **not checked into version control**.

## Instructions for setting up data

1. Place your event data files inside the `dataset/EventsData/events/` folder.
2. Place your user data files inside the `dataset/EventsData/users/` folder.

Example format:
```text
dataset/
├── EventsData/
│   ├── events/
│   │   ├── events_part1.parquet
│   │   └── events_part2.parquet
│   └── users/
│       └── users_data.parquet
```

Once the `.parquet` files are placed inside these directories, the training and ETL pipelines will automatically detect and load them.
