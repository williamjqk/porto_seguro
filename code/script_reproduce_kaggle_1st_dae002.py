# %%
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

# suggest = tpe.suggest
suggest = rand.suggest

# %%
data_root = "/home/tom/data/porto_seguro_dae"
train_file = os.path.join(data_root, "a1a")#"data/a1a"
model_root = os.path.join(data_root, "test_model")#"/tmp/test_model"


extra_param = {
    "train_file": train_file,
    "model_root": model_root,
    "n_val": 2000, # 300
    "summary_step":2, #2, # 这个参数会控制模型文件的输出频率和log打印频率
    "lgb_param": {
        "verbose": -1,
        "lambda_l1": 2 ** sample_int("v_lambda_l1", 0, 2) - 1,
        "lambda_l2": 2 ** sample_int("v_lambda_l2", 0, 2) - 1,
    }
}


from sklearn.datasets import make_classification
X_fake, y_fake = make_classification(n_samples=10000, n_features=500)
from sklearn.datasets import dump_svmlight_file
dump_svmlight_file(X_fake, y_fake, train_file)

import lightgbm as lgb
lgb_model = lgb.train(
    params = {'objective':'binary', 'learning_rate':0.05,  'metric': 'auc'},
    train_set = lgb.Dataset(X_fake[:-extra_param["n_val"],:], y_fake[:-extra_param["n_val"]]),
    valid_sets = (lgb.Dataset(X_fake[-extra_param["n_val"]:,:], y_fake[-extra_param["n_val"]:]),),
    verbose_eval=1,
)

# %%
#proc_run修饰的函数会在一个新的进程中运行，对于像Keras这样的模型，最好在新的进程中运行，以方便对资源进行释放
#注意如果用MongoTrials的时候，优化函数必须是可序列化的（dill），最好不要依赖外部的变量
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
    n_val = param["n_val"]

    X, y = load_svm(file_tr)
    label = y == 1

    X_tr = X[:-n_val]
    label_tr = label[:-n_val]
    X_v = X[-n_val:]
    label_v = label[-n_val:]

    import os
    if not os.path.isdir(model_root):
        os.mkdir(model_root)

    v_metric = ModelMetric(feature_t=X_v, label_t=label_v, name="VA", metrics=["F", "AUC", "KS"], minimize=False)
    tr_metric = ModelMetric(feature_t=X_tr, label_t=label_tr, name="TR", metrics=["f", "auc", "ks"], minimize=False)

    model = BinaryLGB(
        model_root=model_root,
        model_metric=[v_metric, tr_metric], #可以定义多个metric，其中第一个的作为模型选择的基准
        model_name=None # for random model name
    )
    return model.fit(param, X_tr, label_tr)

# %%

space_lgb = space_update(space_lgb_binary, extra_param, model_key="lgb_bin", n_round=10)

# mongo trials可以进行并行训练，可以同时跑多组参数
# from hyperopt.mongoexp import MongoTrials
# trials_lgb = MongoTrials('mongo://$addr/lgb_worker/jobs', exp_key=f'test_model')
trials_lgb = Trials()

from multiprocessing.pool import ThreadPool

with ThreadPool(8) as pool:
    #提交给线程池运行，这样可以同时运行多个fmin
    res = pool.apply_async(fmin, args=(lgb_run, space_lgb), kwds={"trials": trials_lgb, "algo": suggest, "max_evals": 5})
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
for model_path, model_score in select_models(trials_lgb, 3):
    print(-model_score, model_path)
