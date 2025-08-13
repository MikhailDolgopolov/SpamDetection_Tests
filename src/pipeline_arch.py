import inspect
from typing import Any, Dict

from sklearn.pipeline import Pipeline


def _get_default_params(estimator: Any) -> Dict[str, Any]:
    """
    Возвращает словарь параметров по умолчанию для *конструктора* estimator.
    """
    # Получаем конструктор (или любой класс)
    ctor = estimator.__class__
    signature = inspect.signature(ctor.__init__)
    defaults = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty and k != "self"
    }
    return defaults


def _diff_params(estimator: Any) -> Dict[str, Any]:
    """
    Сравнивает актуальные параметры объекта с дефолтами и возвращает только
    изменённые.
    """
    current = estimator.get_params(deep=False)
    default = _get_default_params(estimator)

    # Берём те ключи, которые НЕ равны по умолчанию
    changed = {
        k: v for k, v in current.items() if k not in default or v != default[k]
    }
    return changed


def dump_pipeline_architecture(pipeline: Pipeline) -> Dict[str, Any]:
    arch = {}
    for name, est in pipeline.named_steps.items():
        params = {}
        try:
            params = est.get_params(deep=False)
        except Exception:
            # fallback — небольшая безопасная выборка атрибутов
            params = {
                k: v for k, v in vars(est).items()
                if not k.startswith("_") and not callable(v)
            }
        arch[name] = {"class": est.__class__.__name__, **params}
    return arch
