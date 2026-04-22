-- Feature store schema for UCI Adult Income MLOps pipeline
-- All tables managed by feature_store.py

CREATE TABLE IF NOT EXISTS features_train (
    id          INTEGER PRIMARY KEY,
    age         INTEGER,
    workclass   VARCHAR,
    fnlwgt      INTEGER,
    education   VARCHAR,
    education_num INTEGER,
    marital_status VARCHAR,
    occupation  VARCHAR,
    relationship VARCHAR,
    race        VARCHAR,
    sex         VARCHAR,
    capital_gain INTEGER,
    capital_loss INTEGER,
    hours_per_week INTEGER,
    native_country VARCHAR,
    income      INTEGER
);

CREATE TABLE IF NOT EXISTS features_test (
    id          INTEGER PRIMARY KEY,
    age         INTEGER,
    workclass   VARCHAR,
    fnlwgt      INTEGER,
    education   VARCHAR,
    education_num INTEGER,
    marital_status VARCHAR,
    occupation  VARCHAR,
    relationship VARCHAR,
    race        VARCHAR,
    sex         VARCHAR,
    capital_gain INTEGER,
    capital_loss INTEGER,
    hours_per_week INTEGER,
    native_country VARCHAR,
    income      INTEGER
);

CREATE SEQUENCE IF NOT EXISTS features_inference_seq START 1;

CREATE TABLE IF NOT EXISTS features_inference (
    id              INTEGER PRIMARY KEY DEFAULT nextval('features_inference_seq'),
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    age             INTEGER,
    workclass       VARCHAR,
    education       VARCHAR,
    education_num   INTEGER,
    marital_status  VARCHAR,
    occupation      VARCHAR,
    relationship    VARCHAR,
    race            VARCHAR,
    sex             VARCHAR,
    capital_gain    INTEGER,
    capital_loss    INTEGER,
    hours_per_week  INTEGER,
    native_country  VARCHAR,
    prediction      INTEGER,
    prediction_proba FLOAT,
    model_version   VARCHAR
);
