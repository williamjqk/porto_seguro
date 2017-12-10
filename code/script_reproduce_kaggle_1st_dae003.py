# %% this idea is from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
base_path = '/home/ljc/mywork/some_test/porto_seguro/input/'

model_name = 'porto_seguro_dae003'
data_path = '/home/ljc/data/porto_seguro_dae'
model_path = os.path.join(data_path, model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# %% The last cell preprocess data. In this cell, let data go into one model
# in dae003, try hyperopt
import os
import pandas as pd
import numpy as np
import hyperopt as hp
from hyperopt import fmin, rand, tpe, hp, pyll, base, space_eval, Trials
from hyperopt.pyll import Apply, as_apply, dfs, scope
from hyperopt.mongoexp import MongoTrials
from PPMoney.core.utils import proc_run
from PPMoney.core.model.space import space_lgb_binary, space_update, sample_int
from PPMoney.core.data import HDFDataSet

suggest = rand.suggest

data_root = model_path
model_root = os.path.join(data_root, "test_model")#"/tmp/test_model"


extra_param = {
    "train_file": os.path.join(data_path, 'mjahrer_1st_train.dataset'),
    "model_root": model_root,
    "n_fold": 5,
    "summary_step": 10,
    "lgb_param": {
        "verbose": -1,
        "lambda_l1": 2 ** sample_int("v_lambda_l1", 0, 2) - 1,
        "lambda_l2": 2 ** sample_int("v_lambda_l2", 0, 2) - 1,
    }
}

# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=extra_param['n_fold'], random_state=0)
# train_index, valid_index = next(skf.split(X_0, y_0))

@proc_run
def lgb_run(param):
    import pprint
    import numpy as np
    pprint.pprint(param)
    from sklearn.datasets import load_svmlight_file as load_svm
    from PPMoney.core.model import BinaryLGB
    from PPMoney.core.model.metrics import ModelMetric

    file_tr = param["train_file"]
    model_root = param["model_root"]
    n_fold = param["n_fold"]

    from PPMoney.core.data import HDFDataSet
    dataset_load = HDFDataSet(file_tr, chunk_size=2048)

    X, y = dataset_load['feature'], dataset_load['label']
    label = y == 1
    print(f"X.shape, label.shape: {X.shape, label.shape}")

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_fold, random_state=0)
    train_index, valid_index = next(skf.split(X, label))

    X_tr, X_v = X[train_index], X[valid_index]
    label_tr, label_v = label[train_index], label[valid_index]

    # X_tr = X[:-n_val]
    # label_tr = label[:-n_val]
    # X_v = X[-n_val:]
    # label_v = label[-n_val:]

    import os
    if not os.path.isdir(model_root):
        os.mkdir(model_root)

    v_metric = ModelMetric(feature_t=X_v, label_t=label_v, name="VA", metrics=["AUC"], minimize=False)
    tr_metric = ModelMetric(feature_t=X_tr, label_t=label_tr, name="TR", metrics=["auc"], minimize=False)

    model = BinaryLGB(
        model_root=model_root,
        model_metric=[v_metric, tr_metric], #可以定义多个metric，其中第一个的作为模型选择的基准
        model_name=None # for random model name
    )
    return model.fit(param, X_tr, label_tr)

# 先试试单线程
# fmin(lgb_run, space_lgb, algo=suggest, max_evals=5)


# %%

space_lgb = space_update(space_lgb_binary, extra_param, model_key="lgb_bin", n_round=2000)

# mongo trials可以进行并行训练，可以同时跑多组参数
# from hyperopt.mongoexp import MongoTrials
# trials_lgb = MongoTrials('mongo://$addr/lgb_worker/jobs', exp_key=f'test_model')
trials_lgb = Trials()

from multiprocessing.pool import ThreadPool

with ThreadPool(4) as pool:
    #提交给线程池运行，这样可以同时运行多个fmin
    res = pool.apply_async(fmin, args=(lgb_run, space_lgb), kwds={"trials": trials_lgb, "algo": suggest, "max_evals": 20})
    res.get()

# %%

from hyperopt.plotting import main_plot_vars, main_plot_history, main_plot_histogram
import matplotlib.pylab as plt

for sp, trls in zip([space_lgb], [trials_lgb]):
    domain = base.Domain(lgb_run, sp)
    # plt.figure(figsize=(20, 40))
    # main_plot_vars(trls, bandit=domain, colorize_best=30, columns=1)
    plt.figure(figsize=(20, 5))
    # plt.ylim((-0.003, 0.003))
    main_plot_history(trls, bandit=domain)

# NOTE: 对trials_lgb的性能的评判应该是VA的第一个metric指标的负数，至少在这个例子里是这样

trials_lgb.trials # 所有模型的参数和结果的字典组成的一个list
trials_lgb.results # 返回所有实验的结果
trials_lgb.miscs # 返回所有实验的参数
trials_lgb.vals # 返回所有实验的跟space更新有关的参数

trials_lgb.trials[0]['misc']['vals']
tmp1 = space_lgb['lgb_param']['bagging_fraction']
tmp2 = 2 ** sample_int("v_lambda_l1", 0, 2) - 1
5**-1.6 # 0.07614615754863513

type(tmp2), dir(tmp2)

# NOTE: 试验space到实际数值的映射, 可能和Apply没关系
hp_assignment = {k:v[0] for k,v in trials_lgb.trials[0]['misc']['vals'].items()}
hp_assignment = {k:v[0] for k,v in trials_lgb.vals.items()} # 这句和上面那句等价，这句更简洁
space_eval(space_lgb, hp_assignment)


trials_lgb.trials[0]['result'] # {'loss': -0.8737864077669903, 'status': 'ok'}
# 返回k个最好的模型的参数dict组成的一个list
trials_lgb.topk_trials(k=2)
# return_score=True就返回2个list组成的tuple
trials_lgb.topk_trials(2, return_score=True, ordered=True)
type(trials_lgb.topk_trials(2, return_score=True, ordered=True)[0][0]) # 这个类型就是个dict
# Trials().trial_attachments的作用是，根据trial的参数字典解析出相应的model路径
trials_lgb.trial_attachments(trials_lgb.topk_trials(2, return_score=True, ordered=True)[0][0])["model"].decode()
# %%

#返回topk的模型
select_models = lambda trials, k: [(trials.trial_attachments(t)["model"].decode(), c) for t, c in zip(*trials.topk_trials(k, return_score=True, ordered=True))]
for sub_model_path, sub_model_score in select_models(trials_lgb, 3):
    print(-sub_model_score, sub_model_path)

best_auc = -trials_lgb.topk_trials(1, return_score=True, ordered=True)[1][0]
best_space = trials_lgb.topk_trials(1, return_score=True, ordered=True)[0][0]['misc']['vals']
best_hyperparam = space_eval(space_lgb, hp_assignment = {k:v[0] for k,v in best_space.items()})
best_model_path = select_models(trials_lgb, 1)[0][0]

import json
with open(os.path.join(model_path, 'best_model.json'), 'w') as f:
    json.dump({'best_auc': best_auc, 'best_hyperparam': best_hyperparam, 'best_model_path': best_model_path},
                f, ensure_ascii=False, indent=2, separators=(',', ': '))
