from .base import RelationExtractionDataModule
from .mymodel import MyModelForReDataModule
from ...registry import BaseParent


class AutoReDataModule(BaseParent):

    registry = {}


AutoReDataModule.add_to_registry(MyModelForReDataModule.config_name, MyModelForReDataModule)
