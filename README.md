# MLOps Workshop

General practices.  WIP

### Step 1: Buy apples from farmer

    python honeycrisp/buy_apples.py \
        heartbroken_cucumber \
        data/emails.parquet \
        data/apple_types.parquet \
        data/rejected_mail.parquet \
        data/apples_in_basement.parquet \
        --run-name redhot

### Step 2: Make labels for the Cider bottles

    python honeycrisp/make_labels.py \
        heartbroken_cucumber \
        data/apples_in_basement.parquet \
        data/bottle_labels_embedded.parquet \
        --run-name redhot

### Step 3: Print embeddings graph

    python honeycrisp/print_graph.py \
        heartbroken_cucumber \
        data/bottle_labels_embedded.parquet \
        data/graph.png \
        --run-name redhot
