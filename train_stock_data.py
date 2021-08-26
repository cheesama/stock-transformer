from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling1D, Dropout, Dense
from tqdm import tqdm

from models.time_embedding import Time2Vector
from models.transformer import TransformerEncoder
from models.fastformer import Fastformer
from data.download_stock_data import get_all_stock_data

import argparse

def create_model(seq_len, d_k, d_v, n_heads, ff_dim):
    '''Initialize time and transformer layers'''
    time_embedding = Time2Vector(seq_len)
    attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

    '''Construct model'''
    in_seq = Input(shape=(seq_len, 5)) # open, high, low, close, volume
    x = time_embedding(in_seq)
    x = Concatenate(axis=-1)([in_seq, x])
    x = attn_layer1((x, x, x))
    x = attn_layer2((x, x, x))
    x = attn_layer3((x, x, x))
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(1, activation='linear')(x)

    model = Model(inputs=in_seq, outputs=out)
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])

    return model

def create_model_fastformer(seq_len, dim):
    '''Initialize fastformer layers'''
    fastformer = Fastformer(dim)

    '''Construct model'''
    in_seq = Input(shape=(seq_len, 5)) # open, high, low, close, volume
    x = fastformer(in_seq)
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(1, activation='linear')(x)

    model = Model(inputs=in_seq, outputs=out)
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--ff_dim', type=int, default=512)
    args = parser.parse_args()

    #model = create_model(args.seq_len, args.d_k, args.d_v, args.n_heads, args.ff_dim)
    model = create_model_fastformer(args.seq_len, args.d_k)
    model.summary()

    entire_df = get_all_stock_data()
    print (entire_df.iloc[-1,:])

    for corp_name in tqdm(list(entire_df.Name.unique())):
        if corp_name != '삼성전자':
            continue

        corp_df = entire_df[entire_df.Name == corp_name]
        corp_df = corp_df[['Open', 'High', 'Low', 'Close', 'Volume']].values

        print (corp_df)



    