# neural_cf_music_recommendation.py

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ====== Set seeds for reproducibility ======
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# ====== Generate synthetic dataset ======

# df = pd.read_csv("music_rec_dataset.csv")
num_users = 100
num_items = 500
num_interactions = 1000
latent_dim = 3


# Generate latent factors
user_latent = np.random.normal(size=(num_users, latent_dim))
item_latent = np.random.normal(size=(num_items, latent_dim))

user_ids, item_ids, ratings = [], [], []
for _ in range(num_interactions):
    u = np.random.randint(num_users)
    i = np.random.randint(num_items)
    score = np.dot(user_latent[u], item_latent[i]) + np.random.normal(scale=0.5)
    rating = np.clip(np.round(score + 3), 1, 5)
    user_ids.append(u)
    item_ids.append(i)
    ratings.append(rating)

df = pd.DataFrame({'user_id': user_ids, 'item_id': item_ids, 'rating': ratings})

# ====== Train-test split ======

train, test = train_test_split(df, test_size=0.2, random_state=seed)
train_user = np.array(train['user_id'])
train_item = np.array(train['item_id'])
train_rating = np.array(train['rating'])
test_user = np.array(test['user_id'])
test_item = np.array(test['item_id'])
test_rating = np.array(test['rating'])

# ====== Helper functions for ranking metrics ======

def get_user_pos_test_items(test_df):
    user_pos_items = {}
    for _, row in test_df.iterrows():
        user, item = row['user_id'], row['item_id']
        user_pos_items.setdefault(user, []).append(item)
    return user_pos_items

def hit_ratio_at_k(pred_items, true_items, k):
    pred_k = pred_items[:k]
    return int(any(item in pred_k for item in true_items))

def ndcg_at_k(pred_items, true_items, k):
    pred_k = pred_items[:k]
    for i, pred_item in enumerate(pred_k):
        if pred_item in true_items:
            return 1 / np.log2(i + 2)
    return 0

def evaluate_ranking(model, test_df, train_df, num_users, num_items, K=10):
    user_pos_test = get_user_pos_test_items(test_df)
    hits, ndcgs = [], []
    for user in user_pos_test.keys():
        train_items = set(train_df[train_df['user_id'] == user]['item_id'].values)
        candidates = [i for i in range(num_items) if i not in train_items]
        user_array = np.array([user] * len(candidates))
        item_array = np.array(candidates)
        preds = model.predict([user_array, item_array], batch_size=512, verbose=0).flatten()
        item_score_pairs = sorted(zip(candidates, preds), key=lambda x: x[1], reverse=True)
        ranked_items = [x[0] for x in item_score_pairs]
        true_items = user_pos_test[user]
        hits.append(hit_ratio_at_k(ranked_items, true_items, K))
        ndcgs.append(ndcg_at_k(ranked_items, true_items, K))
    return np.mean(hits), np.mean(ndcgs)

# ====== Build MF baseline model ======

def build_mf_model(num_users, num_items, embedding_size=50):
    user_input = tf.keras.layers.Input(shape=(1,))
    item_input = tf.keras.layers.Input(shape=(1,))
    user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)(user_input)
    item_embedding = tf.keras.layers.Embedding(num_items, embedding_size)(item_input)
    user_vec = tf.keras.layers.Flatten()(user_embedding)
    item_vec = tf.keras.layers.Flatten()(item_embedding)
    mf_output = tf.keras.layers.Dot(axes=1)([user_vec, item_vec])
    model = tf.keras.models.Model(inputs=[user_input, item_input], outputs=mf_output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model



# ====== Build NCF model ======


from tensorflow.keras import regularizers

def build_ncf_model(num_users, num_items, embedding_size=50):
    user_input = tf.keras.layers.Input(shape=(1,))
    item_input = tf.keras.layers.Input(shape=(1,))

    # MF embeddings
    mf_user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)(user_input)
    mf_item_embedding = tf.keras.layers.Embedding(num_items, embedding_size)(item_input)
    mf_user_vec = tf.keras.layers.Flatten()(mf_user_embedding)
    mf_item_vec = tf.keras.layers.Flatten()(mf_item_embedding)
    mf_output = tf.keras.layers.Dot(axes=1)([mf_user_vec, mf_item_vec])

    # MLP embeddings
    mlp_user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)(user_input)
    mlp_item_embedding = tf.keras.layers.Embedding(num_items, embedding_size)(item_input)
    mlp_user_vec = tf.keras.layers.Flatten()(mlp_user_embedding)
    mlp_item_vec = tf.keras.layers.Flatten()(mlp_item_embedding)
    mlp_concat = tf.keras.layers.Concatenate()([mlp_user_vec, mlp_item_vec])

    # MLP layers with L2 regularization and Dropout
    mlp_dense = tf.keras.layers.Dense(128, activation='relu',
                                      kernel_regularizer=regularizers.l2(1e-4))(mlp_concat)
    mlp_dense = tf.keras.layers.Dropout(0.3)(mlp_dense)

    mlp_dense = tf.keras.layers.Dense(64, activation='relu',
                                      kernel_regularizer=regularizers.l2(1e-4))(mlp_dense)
    mlp_dense = tf.keras.layers.Dropout(0.3)(mlp_dense)

    mlp_dense = tf.keras.layers.Dense(32, activation='relu',
                                      kernel_regularizer=regularizers.l2(1e-4))(mlp_dense)
    mlp_dense = tf.keras.layers.Dropout(0.3)(mlp_dense)

    # Combine MF and MLP outputs
    combined = tf.keras.layers.Concatenate()([mf_output, mlp_dense])
    output = tf.keras.layers.Dense(1)(combined)

    model = tf.keras.models.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ====== Train and evaluate models ======

embedding_size = 50

print("Training MF baseline model...")
mf_model = build_mf_model(num_users, num_items, embedding_size)
mf_model.fit([train_user, train_item], train_rating,
             validation_data=([test_user, test_item], test_rating),
             epochs=50, batch_size=256, verbose=2)

mf_loss, mf_mae = mf_model.evaluate([test_user, test_item], test_rating, verbose=0)
# print(f"MF Model Test MAE: {mf_mae:.4f}")
hit_ratio_mf, ndcg_mf = evaluate_ranking(mf_model, test, train, num_users, num_items, K=10)
print(f"MF Model Hit Ratio@10: {hit_ratio_mf:.4f}")
print(f"MF Model NDCG@10: {ndcg_mf:.4f}")

print("\nTraining NCF model...")
ncf_model = build_ncf_model(num_users, num_items, embedding_size)
ncf_model.fit([train_user, train_item], train_rating,
              validation_data=([test_user, test_item], test_rating),
              epochs=50, batch_size=256, verbose=2)

ncf_loss, ncf_mae = ncf_model.evaluate([test_user, test_item], test_rating, verbose=0)
# print(f"NCF Model Test MAE: {ncf_mae:.4f}")
hit_ratio_ncf, ndcg_ncf = evaluate_ranking(ncf_model, test, train, num_users, num_items, K=10)
print(f"NCF Model Hit Ratio@10: {hit_ratio_ncf:.4f}")
print(f"NCF Model NDCG@10: {ndcg_ncf:.4f}")

# ====== Compare improvements ======

hit_ratio_improve = ((hit_ratio_ncf - hit_ratio_mf) / hit_ratio_mf * 100) if hit_ratio_mf > 0 else float('inf')
ndcg_improve = ((ndcg_ncf - ndcg_mf) / ndcg_mf * 100) if ndcg_mf > 0 else float('inf')
print(f"\nHit Ratio Improvement: {hit_ratio_improve:.2f}%")
print(f"NDCG Improvement: {ndcg_improve:.2f}%")

# ====== Inference examples ======

def predict_rating(model, user_id, item_id):
    pred = model.predict([np.array([user_id]), np.array([item_id])], verbose=0)
    return pred[0][0]

def recommend_top_n(model, user_id, train_df, num_items, top_n=10):
    interacted_items = set(train_df[train_df['user_id'] == user_id]['item_id'].values)
    candidates = [i for i in range(num_items) if i not in interacted_items]
    user_array = np.array([user_id] * len(candidates))
    item_array = np.array(candidates)
    preds = model.predict([user_array, item_array], batch_size=512, verbose=0).flatten()
    item_scores = list(zip(candidates, preds))
    item_scores.sort(key=lambda x: x[1], reverse=True)
    top_items = [item for item, score in item_scores[:top_n]]
    return top_items

user_to_predict = 10
item_to_predict = 20
predicted_rating = predict_rating(ncf_model, user_to_predict, item_to_predict)
print(f"Predicted rating for user {user_to_predict} on item {item_to_predict}: {predicted_rating:.4f}")

user_to_recommend = 10
top_items = recommend_top_n(ncf_model, user_to_recommend, train, num_items, top_n=10)
print(f"Top 10 recommended items for user {user_to_recommend}: {top_items}")
