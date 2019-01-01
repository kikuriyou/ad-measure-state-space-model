/*
考慮する成分:
- ローカル線形トレンド成分(1階差分)
- 周期成分
- 回帰成分
*/

data {
    int<lower=0> t_max;                           // 系列長さ
    vector<lower=0>[t_max] y;                     // 目的変数
    vector<lower=0, upper=1>[t_max] x1;    // 回帰成分の説明変数1
    vector<lower=0, upper=1>[t_max] x2;    // 回帰成分の説明変数2
    vector<lower=0, upper=1>[t_max] x3;    // 回帰成分の説明変数3
}

parameters {
    vector[t_max] trend;              // トレンド成分
    vector[t_max] dow;                // 周期成分
    real<lower=0> a1;                 // 回帰成分の係数1
    real<lower=0> a2;                 // 回帰成分の係数2
    real<lower=0> a3;                 // 回帰成分の係数3
    real<lower=0> b;                  // 定数項
    real<lower=0> s_trend;            // トレンド成分の標準偏差
    real<lower=0> s_dow;              // 周期成分の標準偏差
    real<lower=0> s_y;                // 回帰成分の標準偏差
}

transformed parameters {
    vector[t_max] trend_cum;    // トレンド成分の累積
    vector[t_max] pw_impact;    // Point-wise impact
    
    // トレンド成分を累積
    trend_cum[1] = trend[1];
    for (t in 2:t_max){
        trend_cum[t] = trend_cum[t-1] + trend[t];
    }
    
    // Point-wise impact (K. H. Brodersen, 2015)
    pw_impact[1:t_max] = y[1:t_max] - trend_cum[1:t_max] - dow[1:t_max];
}

model {
    // トレンド成分
    for (t in 2:t_max){
        trend[t] ~ normal(trend[t-1], s_trend);                // 1階差分
    }
    
    // 周期成分
    for (t in 7:t_max){
        dow[t] ~ normal(-sum(dow[t-6:t-1]), s_dow);
    }

    // 回帰成分でpoint-wise impactを推測
    pw_impact[1:t_max] ~ normal(a1*x1[1:t_max] + a2*x2[1:t_max] + a3*x3[1:t_max] + b, s_y);
}

generated quantities {
    vector[t_max] y_pred;
    
    // 推定結果を代入
    for (t in 1:t_max){
        y_pred[t] = trend_cum[t] + dow[t] + normal_rng(a1*x1[t] + a2*x2[t] + a3*x3[t] + b, s_y);
    }
}
