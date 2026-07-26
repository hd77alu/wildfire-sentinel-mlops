import os
import sqlite3
import numpy as np
from PIL import Image
import tensorflow as tf
from src.database import DB_PATH

RAW_DIR = "database/retrain_raw"
PROCESSED_MODEL_PATH = "models/mobilenet_wildfire_model.keras"

def preprocess_and_retrain():
    # Fetch unprocessed samples from SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filepath, label FROM retraining_samples WHERE processed = 0")
    records = cursor.fetchall()

    if not records:
        conn.close()
        return {"status": "skipped", "message": "No new dataset samples available for retraining."}

    # Preprocess uploaded images into tensors
    X_list, y_list, processed_ids = [], [], []
    label_map = {"no_wildfire": 0, "wildfire": 1}

    for record_id, filepath, label in records:
        if os.path.exists(filepath) and label.lower() in label_map:
            try:
                img = Image.open(filepath).convert("RGB").resize((224, 224))
                arr = np.array(img, dtype=np.float32) / 255.0  # Rescale
                X_list.append(arr)
                y_list.append(label_map[label.lower()])
                processed_ids.append(record_id)
            except Exception as e:
                print(f"Skipping corrupt image {filepath}: {e}")

    if not X_list:
        conn.close()
        return {"status": "failed", "message": "No valid dataset samples processed."}

    X_train = np.array(X_list)
    y_train = np.array(y_list)

    # Load baseline pretrained model and fine-tune
    model = tf.keras.models.load_model(PROCESSED_MODEL_PATH)
    
    # Re-compile with low learning rate for transfer learning fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=min(8, len(X_train)),
        verbose=1
    )

    # Save updated model checkpoint
    model.save(PROCESSED_MODEL_PATH)

    # Update database record flags
    cursor.executemany("UPDATE retraining_samples SET processed = 1 WHERE id = ?", [(i,) for i in processed_ids])
    cursor.execute(
        "INSERT INTO retraining_jobs (status, samples_used, accuracy) VALUES (?, ?, ?)",
        ("completed", len(processed_ids), float(history.history["accuracy"][-1]))
    )
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "samples_trained": len(processed_ids),
        "final_accuracy": float(history.history["accuracy"][-1])
    }