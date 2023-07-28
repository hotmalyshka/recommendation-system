import os
import pandas as pd
from typing import List, Dict, Union
import psycopg2
from catboost import CatBoostClassifier
from fastapi import FastAPI
from datetime import datetime
from loguru import logger
import hashlib
from schema import PostGet
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()


def get_db():
    db_user = os.environ.get("POSTGRES_USER")
    db_password = os.environ.get("POSTGRES_PASSWORD")
    db_host = os.environ.get("POSTGRES_HOST")
    db_port = os.environ.get("POSTGRES_PORT")
    db_name = os.environ.get("POSTGRES_DATABASE")

    connection_string = f"dbname={db_name} user={db_user} password={db_password} host={db_host} port={db_port}"

    conn = psycopg2.connect(connection_string)
    return conn


def batch_load_sql(query: str):
    with get_db() as conn:
        chunks = []
        for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
            chunks.append(chunk_dataframe)
            logger.info(f"Got chunk: {len(chunk_dataframe)}")
    return pd.concat(chunks, ignore_index=True)


def get_model_path(path: str) -> str:
    # Correct path for model loading
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = f'/workdir/user_input/'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_features():
    # Load liked posts
    logger.info("Loading liked posts")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data 
        where action='like'
    """
    liked_posts = batch_load_sql(liked_posts_query)

    # Load user features
    logger.info("Loading user features")
    user_features_query = """SELECT * FROM public.user_data"""
    user_features = batch_load_sql(user_features_query)

    # Load post features
    logger.info("Loading posts features")
    posts_features_query = """SELECT * FROM public.posts_info_features"""
    posts_features = batch_load_sql(posts_features_query)

    return liked_posts, posts_features, user_features


SALT = "my_salt"  # Salt value
GROUP_PERCENTAGE = 50  # Percentage for grouping


def get_user_group(user_id):
    # Convert user_id to byte string
    user_id_bytes = bytes(str(user_id), encoding='utf-8')

    # Calculate hash value using md5 and salt
    hashed_value = hashlib.md5(user_id_bytes + bytes(SALT, encoding='utf-8')).hexdigest()

    # Convert hash value to decimal number
    hash_number = int(hashed_value, 16)

    # Determine user group  based on percentage splitting
    group = hash_number % 100 < GROUP_PERCENTAGE

    return 'control' if group else 'test'


def load_models(group):
    # Load Catboost model
    model_path = get_model_path("/Users/zirajrsimonan/PycharmProjects/FinalProject2/")

    if group == 'control':
        model_path = os.path.join(model_path, 'model_control')
        loaded_model = CatBoostClassifier()
        loaded_model.load_model(model_path)
        return loaded_model, 'control'
    elif group == 'test':
        model_path = os.path.join(model_path, 'model_test')
        loaded_model = CatBoostClassifier()
        loaded_model.load_model(model_path)
        return loaded_model, 'test'
    else:
        raise ValueError("Invalid group value. Must be 'control' or 'test'.")


def get_recommended_feed(id: int, time: datetime, limit: int) -> Dict[str, Union[str, List[PostGet]]]:
    group = get_user_group(id)
    logger.info(f"User {id} belongs to group {group}")
    model, model_type = load_models(group)

    liked_posts, posts_features, user_features = load_features()
    features = (liked_posts, posts_features, user_features)
    liked_posts.fillna("NA", inplace=True)
    user_features.fillna("NA", inplace=True)
    posts_features.fillna("NA", inplace=True)

    user_features = user_features[user_features['user_id'] == id]
    user_features = user_features.drop('user_id', axis=1)

    posts_features = posts_features.drop(['index', 'text', 'topic'], axis=1)
    content = features[1][['post_id', 'text', 'topic']]  # Update the column names

    user_posts_features = pd.merge(posts_features, user_features, how='left', left_index=True, right_index=True)

    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    user_posts_features.fillna("NA", inplace=True)

    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features["predicts"] = predicts

    liked_posts = liked_posts[liked_posts['user_id'] == id].post_id.values
    filtered_posts = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    recommended_posts = filtered_posts.sort_values('predicts')[-limit:].index

    recommended_posts_data = content[content['post_id'].isin(recommended_posts)].reset_index(drop=True)
    recommended_posts_list = []
    for _, row in recommended_posts_data.iterrows():
        recommended_post = PostGet(
            id=row['post_id'],
            text=row['text'],
            topic=row['topic']
        )
        recommended_posts_list.append(recommended_post)

    return {
        'exp_group': group,
        'model_used': model_type,
        'recommended_posts': recommended_posts_list
    }


@app.get("/post/recommendations/", response_model=Dict[str, Union[str, List[PostGet]]])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> Dict[str, Union[str, List[PostGet]]]:
    return get_recommended_feed(id, time, limit)