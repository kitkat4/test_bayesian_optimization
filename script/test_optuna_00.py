
#coding:utf-8

import optuna

# 目的関数の名を冠しているが、実際には本当の意味での目的関数の計算機能だけでなく、
# そのトライアルで目的関数に与える入力を生成する機能も備えるようだ。
# Optunaを使った最適化において、こうした関数はTrialクラスのオブジェクトを引数に取る必要がある。
def objective(trial):
    # suggest_floatは、目的関数に対する入力を生成させるメソッド。
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)

    # 目的関数を計算
    return 2*x**2 - 1.05*x**4 + x**6/6 + x*y + y**2

if __name__ == '__main__':
    
    # どんな最適化をしたいかをStudyクラスのオブジェクトとして定義する。
    # デフォルトではTPEによる最小化が選択される。
    # 最大化したいならdirection="maximize"とする。
    # TPEではなくガウス過程に基づくベイズ最適化を行いたい場合は、
    # sampler=optuna.integration.BoTorchSampler()を指定する。
    study = optuna.create_study(sampler=optuna.integration.BoTorchSampler())

    # 最適化を実行する。結果はstudyオブジェクトに格納される。
    study.optimize(objective, n_trials = 100)

    print(
        f"Best value: {study.best_value} "
        f"(params: {study.best_params})"
    )
    
