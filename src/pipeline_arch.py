import inspect
from typing import Any, Dict


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


def dump_pipeline_architecture(pipeline) -> Dict[str, Any]:
    """
    Возвращает «чистую» схему пайплайна.
    Если объект – Pipeline, то возвращаем список шагов с их изменёнными
    параметрами. Если не Pipeline – просто вернём его имя и параметры.
    """
    if hasattr(pipeline, "steps"):  # это sklearn.pipeline.Pipeline
        steps_info = []
        for name, est in pipeline.steps:
            steps_info.append(
                {
                    type(est).__name__: _diff_params(est),
                }
            )
        return {"Pipeline": steps_info}
    else:  # обычный Estimator
        return {type(pipeline).__name__: _diff_params(pipeline)}
