try:
    import wandb as _wandb
    _wandb_import_error = None
except Exception as exc:
    _wandb = None
    _wandb_import_error = exc


class _WandbTableStub:
    def __init__(self, columns=None):
        self.columns = columns or []
        self.rows = []

    def add_data(self, *args):
        self.rows.append(args)


class _WandbStub:
    Table = _WandbTableStub

    def init(self, *args, **kwargs):
        return None

    def define_metric(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None


wandb = _wandb if _wandb is not None else _WandbStub()


def wandb_available():
    return _wandb is not None


def wandb_import_error():
    return _wandb_import_error
