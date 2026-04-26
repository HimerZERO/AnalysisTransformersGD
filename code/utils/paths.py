from pathlib import Path

def get_project_root():
    """
    Возвращает абсолютный путь к корню проекта.
    Определяется относительно местоположения ЭТОГО файла.
    """
    return Path(__file__).parent.parent


def add_project_root_to_sys_path():
    """
    Добавляет корень проекта в sys.path, если его там ещё нет.
    """
    import sys
    root = str(get_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)