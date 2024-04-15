from src.models.ModelBaseClass import ModelBaseClass
from src.models.WRENCHModels.DawidSkeneModel import DawidSkeneModel
from src.models.WRENCHModels.SnorkelModels import SnorkelLabelModel, SnorkelMajorityLabelVoter, SnorkelModelLoader
from src.models.LGBModels import LGBMSKLearnModel, LGBMSKLearnModel2, LGBMSKLearnModel3
from src.models.SKLearnModels.SkLearnModels import instantiate_sk_learn_models
from src.models.PyTorchModels.PyTorchModels import load_pytorch_models

def load_models(model_config):
    models_in_use = []

    if model_config['models_in_use']["use_dawid_skene_models"]:
        models_in_use.append(DawidSkeneModel())

    if model_config['models_in_use']["use_sklearn_models"]:
        models_in_use.extend(instantiate_sk_learn_models())

    if model_config['models_in_use']["use_snorkel_models"]:
        snorkel_label_model_loader = SnorkelModelLoader()
        snorkel_models = snorkel_label_model_loader.load_snorkel_models()
        models_in_use.extend([SnorkelMajorityLabelVoter()])
        models_in_use.extend(snorkel_models)

    if model_config['models_in_use']["use_lgb_models"]:
        models_in_use.extend([LGBMSKLearnModel3(), LGBMSKLearnModel(), LGBMSKLearnModel2()])

    if model_config['models_in_use']["use_pytorch_models"]:
        pytorch_models = load_pytorch_models()
        models_in_use.extend(pytorch_models)

    if model_config['models_in_use']["use_GAM_models"]:
        from src.models.GAMModels import LogisticGAMModel, LogisticGAMModel2
        models_in_use.extend([LogisticGAMModel()])

    if model_config['models_in_use']["use_ml_lense"]:
        raise ValueError("ML Lense is not currently supported with Python 3.11.")

    return models_in_use

