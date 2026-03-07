import os
from datetime import datetime
import gc
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from keras.utils import to_categorical
from sklearn.utils import class_weight

from lib.check_score import check_score_and_save
from lib.defensive_distiller_heatmap import Defensive_Distiller_heatmap
from lib.distiller_heatmap import Distiller_heatmap
from lib.load_student_model import load_student
from lib.set_dashboard import set_dashboard_distiller
from lib.split import split
from lib.standard_distiller import Standard_Distiller
from lib.student_compile import student_compile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from lib.custom_attention import attention_map_no_norm_fast
import pandas as pd
from keras import backend as K

tf.data.experimental.enable_debug_mode()
tf.config.run_functions_eagerly(True)
config_tf = tf.compat.v1.ConfigProto()
config_tf.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

os.environ['HYPEROPT_FMIN_SEED'] = "0"

session = InteractiveSession(config=config_tf)

score_list = []
best_loss = None
now = datetime.now()
date = now.strftime("%Y-%m-%d-%H-%M-%S")
i = -1

best_model = None
best_score = None


def create_heatmap(teacher, data, split_rate, all=False):
    im = []
    index = list(split(range(len(data)), split_rate))
    for k in index:
        im1, _, _ = attention_map_no_norm_fast(teacher, data[k])
        im.extend(im1)
    im = np.array(im)
    return im


def create_ds_with_heatmap(data, im):
    x_with_h = np.array([data, im])
    x_with_h = np.swapaxes(x_with_h, 0, 1)
    return x_with_h


def hyperopt_loop(param):

    gc.collect()
    
    config = param["config"]
    teacher = param["teacher"]
    y_train = param["y_train"]
    y_val = param["y_val"]
    y_test = param["y_test"]
    x_with_h = param["x_with_h"]
    x_with_h_val = param["x_with_h_val"]
    x_with_h_test = param["x_with_h_test"]

    global i, best_model, best_score
    i = i + 1
    shape = x_with_h.shape[2:5]
    start = datetime.now()
    student = load_student(config, shape, len(set(y_train)), param)
    student_compile(student, param)

    distiller = Distiller_heatmap(student=student, teacher=teacher)

    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=param["A"],
        temperature=param["T"],
    )

    callbacks = []
    if config.getboolean("DISTILLATION", "EarlyStop"):
        print("Early stop in HyperOpt")
        stop_callback = tf.keras.callbacks.EarlyStopping(monitor='student_loss', min_delta=0.0001,
                                                         patience=config.getint("DISTILLATION", "Patience"),
                                                         restore_best_weights=True, verbose=2)
        callbacks.append(stop_callback)

    one_hot_encode_y_train = to_categorical(y_train, num_classes=len(set(y_train)))
    one_hot_encode_y_val = to_categorical(y_val, num_classes=len(set(y_test)))

    history = distiller.fit(
            x=x_with_h,
            y=one_hot_encode_y_train,
            batch_size=param["batch"],
            epochs=config.getint("DISTILLATION", "Epochs"),
            validation_data=(x_with_h_val, one_hot_encode_y_val),
            verbose=1,
            callbacks=callbacks,
    )

    # distiller.evaluate(x_with_h, y_test)
    # scores, res_test = check_score_and_save(history, distiller, x_with_h, y_train, x_with_h_val, y_val, x_with_h_test,
    #                                         y_test,
    #                                         config, save=False, distillation=True, dashboard=wandb,
    #                                         time=datetime.now() - start)

    scores = param.copy()
    scores.update(history.history)
    score_list.append(scores)

    global best_loss
    if best_loss is None:
        best_loss = score_list[-1]["val_student_loss"][0]
        best_model = distiller
        best_score = scores

    elif score_list[-1]["val_student_loss"][0] < best_loss:
        best_loss = score_list[-1]["val_student_loss"][0]
        best_model = distiller
        best_score = scores

    K.clear_session()

    return {'loss': scores["val_student_loss"][0], 'status': STATUS_OK}


def VINCENT_fit(config_g, teacher_g, x_with_h_g, y_train_g, x_with_h_val_g, y_val_g, x_with_h_test_g, y_test_g):

    x_with_h = x_with_h_g
    y_train = y_train_g
    x_with_h_val = x_with_h_val_g
    y_val = y_val_g
    x_with_h_test = x_with_h_test_g
    y_test = y_test_g
    teacher = teacher_g
    config = config_g

    trials = Trials()
    optimizable_variable = {
        "config": config,
        "teacher": teacher,
        "y_val": y_val,
        "y_train": y_train,
        "y_test": y_test,
        "x_with_h": x_with_h,
        "x_with_h_val": x_with_h_val,
        "x_with_h_test": x_with_h_test,
        "kernel": hp.choice("kernel", np.arange(2, 3 + 1)),
        "batch": hp.choice("batch", [64, 128, 256, 512]), #, 512
        'dropout1': hp.uniform("dropout1", 0, 1),
        'dropout2': hp.uniform("dropout2", 0, 1),
        "learning_rate": hp.uniform("learning_rate", 1e-4, 1e-3),
        "T": hp.choice("T", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "A": hp.uniform("A", 0, 1)
    }

    fmin(hyperopt_loop, optimizable_variable, trials=trials, algo=tpe.suggest,
         max_evals=config.getint("DISTILLATION", "HyperoptEvaluations"))

    return best_model, best_score
