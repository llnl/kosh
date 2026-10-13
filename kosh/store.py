import os
import gc
import atexit
import sys
import uuid
import collections
if not sys.platform.startswith("win"):
    import fcntl
    import grp
import hashlib
import warnings
import time
from datetime import datetime
import re
from .loaders import KoshFileLoader, PGMLoader, KoshSinaLoader
from .utils import compute_fast_sha, merge_datasets_handler, __check_valid_connection_type__, _update_record
from .loaders import JSONLoader
from .loaders import NpyLoader
from .loaders import NumpyTxtLoader
from .loaders import PandasLoader
from .dataset import KoshDataset
from .ensemble import KoshEnsemble
from .core_sina import KoshSinaFile, KoshSinaObject, kosh_pickler
from .utils import create_kosh_users
from .utils import update_store_and_get_info_record, get_store_info_record
from .utils import get_store_info_record_attribute
from inspect import isfunction, ismethod
import kosh
import six
import types
import sys
from .kosh_command import KoshCmd, process_cmd
from . import lock_strategies
try:
    from .loaders import HDF5Loader
except ImportError:
    pass
try:
    from .loaders import PILLoader
except ImportError:
    pass
try:
    from .loaders import UltraLoader
except ImportError:
    pass
try:
    from .loaders import SidreMeshBlueprintFieldLoader
except ImportError:
    pass
try:
    import orjson
except ImportError:
    import json as orjson  # noqa


@lock_strategies.lock_function
def connect(database, keyspace=None, database_type=None,
            allow_connection_pooling=False, read_only=False,
            delete_all_contents=False, execution_options={},
            connection_type="write", lock_strategy=None,
            verbose=False, **kargs):
    """Connect to a Sina store.

Given a uri/path (and, if required, the name of a keyspace),
figures out which backend is required.

:param database: The URI of the store to connect to.
:type database: str
:param keyspace: The keyspace to connect to (Cassandra only).
:type keyspace: str
:param database_type: Type of backend to connect to. If not provided, Sina
                      will infer this from <database>. One of "sql" or
                      "cassandra".
:type database_type: str
:param allow_connection_pooling: Allow "pooling" behavior that recycles connections,
                                 which may prevent them from closing fully on .close().
                                 Only used for the sql backend.
:type allow_connection_pooling: bool
:param read_only: whether to create a read-only store
:type read_only: bool
:param execution_options: execution options keyword to pass to sina store record_dao at creation time
:type execution_options: dict
:param connection_type: whether to create a write, append, or read only store
:type connection_type: str
:param kargs: Any extra arguments you wish to pass to the KoshStore function
:type kargs: dict, key=value
:param delete_all_contents: Deletes all data after opening the db
:type delete_all_contents: bool
:param lock_strategy: A LockStrategy object to incorporate locking mechanisms
:type lock_strategy: LockStrategy
:return: a KoshStore object connected to the specified database
:rtype: KoshStoreClass
"""
    from sina import connect as sina_connect
    from sqlalchemy.exc import ResourceClosedError
    if lock_strategy is None:
        lock_strategy = lock_strategies.NoLocking()

    with lock_strategy:
        db = kargs.pop("db", None)
        if db is not None:
            if database_type is not None and db != database_type:
                raise ValueError(
                    "You cannot specify `db` and `database_type` with different values")
            database_type = db
        if connection_type not in ['write', 'append', 'read']:
            raise ValueError(
                "`connection_type` must be one of ['write', 'append', 'read']")
        if connection_type == 'read':
            read_only = True
        sina_store = sina_connect(database=database,
                                  keyspace=keyspace,
                                  database_type=database_type,
                                  allow_connection_pooling=allow_connection_pooling,
                                  read_only=read_only, connection_type=connection_type)
        # sina_store._record_dao.session.connection(execution_options=execution_options)

        if not read_only and connection_type == 'write':
            if delete_all_contents:
                try:
                    sina_store.delete_all_contents(force="SKIP PROMPT")
                except ResourceClosedError:
                    pass  # Store was already empty
            update_store_and_get_info_record(sina_store.records)
            create_kosh_users(sina_store.records)
        sina_store.close()
        sync = kargs.pop("sync", True)
        if read_only:
            sync = False
        store = KoshStore(database, sync=sync, keyspace=keyspace, read_only=read_only,
                          db=database_type,
                          allow_connection_pooling=allow_connection_pooling,
                          execution_options=execution_options, connection_type=connection_type,
                          lock_strategy=lock_strategy,
                          verbose=verbose,
                          **kargs)
        return store


class KoshStore(object):
    """Kosh store, relies on Sina"""

    def __init__(self, db_uri=None, username=os.environ.get("USER", "default"), db=None,
                 keyspace=None, sync=True, dataset_record_type="dataset",
                 verbose=True, use_lock_file=False, kosh_reserved_record_types=[],
                 read_only=False, allow_connection_pooling=False, ensemble_predicate=None,
                 execution_options={}, connection_type='write', lock_strategy=None,
                 session_sort_by=None, session_sort_by_descending=False, verbose_attributes=False):
        """__init__ initialize a new Sina-based store

        :param db: type of database, defaults to 'sql', can be 'cass'
        :type db: str, optional
        :param username: user name defaults to user id
        :type username: str
        :param db_uri: uri to sql file or list of cassandra node ips, defaults to None
        :type db_uri: str or list, optional
        :param keyspace: cassandra keyspace, defaults to None
        :type keyspace: str, optional
        :param sync: Does Kosh sync automatically to the db (True) or on demand (False)
        :type sync: bool
        :param dataset_record_type: Kosh element type is "dataset" this can change the default
                                    This is useful if reading in other sina db
        :type dataset_record_type: str
        :param verbose: verbose message
        :type verbose: bool
        :param use_lock_file: If you receive sqlite threads access error, turning this on might help
        :type use_lock_file: bool
        :param kosh_reserved_record_types: list of record types that are reserved for Kosh internal
                                           use, will be ignored when searching store
        :type kosh_reserved_record_types: list of strings
        :param read_only: Can we modify the database source?
        :type read_only: bool
        :param allow_connection_pooling: Allow "pooling" behavior that recycles connections,
                                        which may prevent them from closing fully on .close().
                                        Only used for the sql backend.
        :type allow_connection_pooling: bool
        :param ensemble_predicate: The predicate for the relationship to an ensemble
        :type ensemble_predicate: str
        :param execution_options: execution options keyword to pass to sina store record_dao at creation time
        :type execution_options: dict
        :param connection_type: whether to create a write, append, or read only store
        :type connection_type: str
        :param lock_strategy: The LockStrategy to apply to class methods
        :type lock_strategy: LockStrategy
        :raises ConnectionRefusedError: Could not connect to cassandra
        :raises SystemError: more than one user match.
        :param session_sort_by: A default sort_by key for find operations for this store
        :type session_sort_by: str
        :param session_sort_by_descending: A default sort_by_descending flag for this store
        :type session_sort_by_descending: bool
        :param verbose_attributes: Should we print attribute values of over length 30?
        :type verbose_attributes: bool
        """
        from sina import connect as sina_connect
        from sina.utils import Negation
        if lock_strategy is None:
            lock_strategy = lock_strategies.NoLocking()
        self.lock_strategy = lock_strategy
        self._closed = False
        self._atexit_cleanup = None
        self.session_sort_by = session_sort_by
        self.session_sort_by_descending = session_sort_by_descending
        self.verbose_attributes = verbose_attributes

        with lock_strategy:
            if db_uri is not None and "://" in db_uri and use_lock_file:
                warnings.warn("You cannot use `lock_file` on non file-based db, turning it off", ResourceWarning)
                use_lock_file = False
            if connection_type not in ['write', 'append', 'read']:
                raise ValueError(
                        "`connection_type` must be one of ['write', 'append', 'read']")
            self.__connection_type__ = connection_type
            self.use_lock_file = use_lock_file
            if connection_type == 'read':
                read_only = True
            if read_only:
                sync = False
            self.__read_only__ = read_only
            self.__sync__ = sync
            self.__sync__dict__ = {}
            self.__sync__deleted__ = {}

            if db is None:
                db = 'sql'
            self._dataset_record_type = dataset_record_type
            self.db_uri = db_uri
            if db == "sql":
                if db_uri is not None and not os.path.exists(db_uri):
                    if "://" in db_uri:
                        self.__sina_store = sina_connect(
                            db_uri, read_only=read_only, connection_type=connection_type)
                        self.__sina_store._record_dao.session.connection(execution_options=execution_options)
                    else:
                        raise ValueError(
                            "Kosh store could not be found at: {}".format(db_uri))
                else:
                    self.lock()
                    if db_uri is not None:
                        db_pth = os.path.abspath(db_uri)
                    else:
                        db_pth = None
                    self.__sina_store = sina_connect(database=db_pth,
                                                     read_only=read_only,
                                                     database_type=db,
                                                     allow_connection_pooling=allow_connection_pooling,
                                                     connection_type=connection_type)
                    self.__sina_store._record_dao.session.connection(execution_options=execution_options)
                    self.unlock()
            elif db.lower().startswith('cass'):
                self.__sina_store = sina_connect(
                    keyspace=keyspace, database=db_uri,
                    database_type='cassandra', read_only=read_only,
                    allow_connection_pooling=allow_connection_pooling,
                    connection_type=connection_type)

            rec = update_store_and_get_info_record(self.__sina_store.records, ensemble_predicate)

            self._cached_features_ = rec["user_defined"]["cached_features"]
            self._sources_type = rec["data"]["sources_type"]["value"]
            self._users_type = rec["data"]["users_type"]["value"]
            self._groups_type = rec["data"]["groups_type"]["value"]
            self._loaders_type = rec["data"]["loaders_type"]["value"]
            self._ensembles_type = rec["data"]["ensembles_type"]["value"]
            self._ensemble_predicate = rec["data"]["ensemble_predicate"]["value"]
            self._kosh_reserved_record_types = kosh_reserved_record_types + \
                rec["data"]["reserved_types"]["value"]
            kosh_reserved = list(self._kosh_reserved_record_types)
            kosh_reserved.remove(self._sources_type)
            self._kosh_datasets_and_sources = Negation(kosh_reserved)

            # Associated stores
            self._associated_stores_ = []
            if "associated_stores" in rec["data"]:
                for store in rec["data"]["associated_stores"]["value"]:
                    try:
                        self._associated_stores_.append(kosh.connect(store, read_only=read_only, sync=sync,
                                                                     connection_type=connection_type))
                    except Exception:  # most likely a sqlalchemy.exc.DatabaseError
                        warnings.warn("Could not open associated store: {}".format(store))

            self.lock()
            self.__dict__["__record_handler__"] = self.__sina_store.records
            self.unlock()
            users_filter = list(self.__record_handler__.find_with_type(
                self._users_type, ids_only=True))
            names_filter = list(
                self.__record_handler__.find_with_data(
                    username=username))
            inter_recs = set(users_filter).intersection(set(names_filter))
            if len(inter_recs) == 0:
                # raise ConnectionRefusedError("Unknown user: {}".format(username))
                # For now just letting anyone log in as anonymous
                warnings.warn("Unknown user, you will be logged as anonymous user")
                names_filter = self.__record_handler__.find_with_data(
                    username="anonymous")
                self.__user_id__ = "anonymous"
            elif len(inter_recs) > 1:
                raise SystemError("Internal error, more than one user match!")
            else:
                self.__user_id__ = list(inter_recs)[0]
            self.storeLoader = KoshSinaLoader
            self.loaders = {}
            self.add_loader(self.storeLoader)

            # self.storeLoader = KoshLoader
            self.add_loader(KoshFileLoader)
            self.add_loader(JSONLoader)
            self.add_loader(NpyLoader)
            self.add_loader(NumpyTxtLoader)
            self.add_loader(PandasLoader)
            try:
                self.add_loader(HDF5Loader)
            except Exception:  # no h5py module?
                if verbose:
                    warnings.warn("Could not add hdf5 loader, check if you have h5py installed."
                                  " Pass verbose=False when creating the store to turn this message off")
            try:
                self.add_loader(PILLoader)
            except Exception:  # no PIL?
                if verbose:
                    warnings.warn("Could not add pil loader, check if you have pillow installed."
                                  " Pass verbose=False when creating the store to turn this message off")
            self.add_loader(PGMLoader)
            try:
                self.add_loader(UltraLoader)
            except Exception:  # no pydv?
                if verbose:
                    warnings.warn("Could not add ultra files loader, check if you have pydv installed."
                                  " Pass verbose=False when creating the store to turn this message off")
            try:
                self.add_loader(SidreMeshBlueprintFieldLoader)
            except Exception:  # no conduit?
                if verbose:
                    warnings.warn("Could not add sidre blueprint meshfield loader, check if you have conduit installed."
                                  " Pass verbose=False when creating the store to turn this message off")
            # Now let's add the loaders in the store
            for rec_loader in self.__record_handler__.find_with_type("koshloader"):
                loader = kosh_pickler.loads(rec_loader.data["code"]["value"])
                self.add_loader(loader)
            self._added_unsync_mem_store = sina_connect(None)
            self._cached_loaders = collections.OrderedDict()
            # self._cached_features = "cached_features"

            # Ok we need to map the KoshFileLoader back to whatever the source_type is
            # in this store
            ks = self.loaders["file"]
            for loader in ks:
                loader.types[self._sources_type] = loader.types["file"]
            self.loaders[self._sources_type] = self.loaders["file"]
            # Keep the shutdown hook tied to the live store instance so an
            # explicit close can unregister it before interpreter teardown.
            self._atexit_cleanup = self._clean_up
            atexit.register(self._atexit_cleanup)

    @lock_strategies.lock_method
    def __enter__(self):
        return self

    @lock_strategies.lock_method
    def add_loader(self, loader, save=False):
        """Adds a loader to the store

        :param loader: The Kosh loader you want to add to the store
        :type loader: KoshLoader
        :param save: Do we also save it in store for later re-use
        :type save: bool

        :return: None
        :rtype: None
        """
        __check_valid_connection_type__(self.__connection_type__, ['write', 'append'])
        # We add a loader we need to clear the cache
        self._cached_loaders = collections.OrderedDict()

        for k in loader.types:
            if k in self.loaders:
                if loader not in self.loaders[k]:
                    self.loaders[k].append(loader)
            else:
                self.loaders[k] = [loader, ]

        if save:  # do we save it in store
            self.save_loader(loader)

    @lock_strategies.lock_method
    def delete_loader(self, loader, permanently=False):
        """Removes a loader from the store and possible from its db

        :param loader: The Kosh loader you want to remove
        :type loader: KoshLoader
        :param permanently: Do we also remove it from the db if saved there?
        :type permanently: bool

        :return: None
        :rtype: None
        """
        __check_valid_connection_type__(self.__connection_type__, ['write'])
        # We deleted a loader we need to clear the cache
        existing_cached = self._cached_loaders.values()
        empty_types = []
        for k in loader.types:
            if k in self.loaders:
                if loader in self.loaders[k]:
                    self.loaders[k].remove(loader)
                    if len(self.loaders[k]) == 0:
                        # ok we took them all out
                        empty_types.append(k)
                    __listed_features_cache = self._cached_features_
                    yank = []
                    for id in __listed_features_cache.keys():
                        for ld, _ in existing_cached:
                            if isinstance(ld, loader) and id.startswith(ld.signature):
                                yank.append(id)
                    for id in yank:
                        del __listed_features_cache[id]
                    self._cached_features_ = __listed_features_cache
        for k in empty_types:
            del self.loaders[k]
        self._cached_loaders = collections.OrderedDict()

        if permanently:  # Remove it from saved in db as well
            pickled = kosh_pickler.dumps(loader)
            rec = next(self.find(types="koshloader", code=pickled, ids_only=True), None)
            if rec is not None:
                _update_record(rec, self, delete=True)

    @lock_strategies.lock_method
    def remove_loader(self, loader):
        """Removes a loader from the store and its db

        :param loader: The Kosh loader you want to add to the store
        :type loader: KoshLoader

        :return: None
        :rtype: None
        """
        __check_valid_connection_type__(self.__connection_type__, ['write'])
        self.delete_loader(loader, permanently=True)

    @lock_strategies.lock_method
    def lock(self):
        """Attempts to lock the store, helps when many concurrent requests are made to the store"""
        if not self.use_lock_file or "://" in self.db_uri:
            return
        locked = False
        while not locked:
            try:
                self.lock_file = open(self.db_uri + ".handle", "w")
                fcntl.lockf(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except Exception:
                time.sleep(0.1)

    @lock_strategies.lock_method
    def unlock(self):
        """Unlocks the store so other can access it"""
        if not self.use_lock_file or "://" in self.db_uri:
            return
        fcntl.lockf(self.lock_file, fcntl.LOCK_UN)
        self.lock_file.close()
        # Wrapping this in a try/except
        # In case concurrency by same user
        # already removed the file
        try:
            os.remove(self.lock_file.name)
        except Exception:
            pass

    @lock_strategies.lock_method
    def _clean_up(self):
        """delete the KoshStore object"""
        self.close()
        if not self.use_lock_file or "://" in self.db_uri:
            return
        name = self.lock_file.name
        self.lock_file.close()
        if os.path.exists(name):
            os.remove(name)

    @lock_strategies.lock_method
    def get_sina_store(self):
        """Returns the sina store object"""
        return self.__sina_store

    @lock_strategies.lock_method
    def get_sina_records(self):
        """Returns sina store's records"""
        return self.__sina_store.records

    @lock_strategies.lock_method
    def close(self):
        """closes store and sina related things"""
        if self._closed:
            return
        self._closed = True
        cleanup = self._atexit_cleanup
        self._atexit_cleanup = None
        try:
            if cleanup is not None:
                atexit.unregister(cleanup)
        except Exception:
            pass
        try:
            # We need to update the features
            recs = self.get_sina_records()
            store_rec = get_store_info_record(recs)
            cached_features = get_store_info_record_attribute(recs, "cached_features")
            cached_features.update(self._cached_features_)
            store_rec["user_defined"]["cached_features"] = cached_features
            _update_record(store_rec, self)
        except Exception:  # store is likely closed already
            pass
        # Best-effort cleanup to avoid SQLAlchemy emitting
        # "SQLite objects created in a thread..." warnings at interpreter shutdown.
        # These can happen if a DB-API connection is finalized on a different
        # thread than it was created on.
        try:
            dao = getattr(self.__sina_store, "_record_dao", None)
            session = getattr(dao, "session", None) if dao is not None else None
            if session is not None:
                try:
                    bind = session.get_bind()
                except Exception:
                    bind = None
                session.close()
                if bind is not None and hasattr(bind, "dispose"):
                    bind.dispose()
        except Exception:
            pass
        self.__sina_store.close()
        gc.collect()

    @lock_strategies.lock_method
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    @lock_strategies.lock_method
    def delete_all_contents(self, force=""):
        """
        Delete EVERYTHING in a datastore; this cannot be undone.

        :param force: This function is meant to raise a confirmation prompt. If you
                      want to use it in an automated script (and you're sure of
                      what you're doing), set this to "SKIP PROMPT".
        :type force: str
        :returns: whether the deletion happened.
        """
        __check_valid_connection_type__(self.__connection_type__, ['write'])
        ret = self.__sina_store.delete_all_contents(force=force)
        update_store_and_get_info_record(self.__sina_store.records)
        create_kosh_users(self.__sina_store.records)
        return ret

    @lock_strategies.lock_method
    def save_loader(self, loader):
        """Save a loader to the store
        Executed immediately even in async mode

        :param loader: Loader to save
        :type loader: KoshLoader
        """
        from sina.model import Record
        __check_valid_connection_type__(self.__connection_type__, ['write', 'append'])
        pickled = kosh_pickler.dumps(loader)
        rec = next(self.find(types="koshloader", code=pickled, ids_only=True), None)
        if rec is not None:
            # already in store
            return
        rec = Record(id=uuid.uuid4().hex, type="koshloader", user_defined={'kosh_information': {}})
        rec.add_data("code", pickled)
        _update_record(rec, self)

    @lock_strategies.lock_method
    def get_record(self, Id):
        """Gets the sina record tied to an id
        tags record with time of last access to db
        :param Id: record id
        :type Id: str
        :return: sina record
        :rtype: sina.model.Record
        """
        if (not self.__sync__) and Id in self.__sync__dict__:
            record = self.__sync__dict__[Id]
        else:
            record = self.__record_handler__.get(Id)
            self.__sync__dict__[Id] = record
            if not self.__sync__:  # we are not autosyncing
                try:
                    keys = list(record["user_defined"]['kosh_information'].keys())
                except KeyError:
                    record["user_defined"]['kosh_information'] = {}
                    keys = []
                for key in keys:
                    if key[-14:] == "_last_modified":
                        del record["user_defined"]['kosh_information'][key]

        return record

    @lock_strategies.lock_method
    def delete(self, Id):
        """remove a record from store.
        for datasets dissociate all associated data first.

        :param Id: unique Id or kosh_obj
        :type Id: str
        """
        __check_valid_connection_type__(self.__connection_type__, ['write'])
        if not isinstance(Id, six.string_types):
            Id = Id.id
        rec = self.get_record(Id)
        if rec.type not in self._kosh_reserved_record_types:
            kosh_obj = self.open(Id)
            for uri in list(rec["files"].keys()):
                # Let's dissociate to remove unused kosh objects as well
                kosh_obj.dissociate(uri)
        if not self.__sync__:
            _update_record(Id, self, self._added_unsync_mem_store, delete=True)
            if Id in self.__sync__dict__:
                del self.__sync__dict__[Id]
                self.__sync__deleted__[Id] = rec
                rec["user_defined"]['kosh_information']["deleted_time"] = time.time()
        else:
            _update_record(Id, self, delete=True)

    @lock_strategies.lock_method
    def create_ensemble(self, name="Unnamed Ensemble", id=None, metadata={}, schema=None, **kargs):
        """Create a Kosh ensemble object
        :param name: name for the dataset, defaults to None
        :type name: str, optional
        :param id: unique Id, defaults to None which means use uuid4()
        :type id: str, optional
        :param metadata: dictionary of attribute/value pair for the dataset, defaults to {}
        :type metadata: dict, optional
        :param schema: a KoshSchema object to validate datasets and when setting attributes
        :type schema: KoshSchema
        :param kargs: extra keyword arguments (ignored)
        :type kargs: dict
        :raises RuntimeError: Dataset already exists
        :return: KoshEnsemble
        :rtype: KoshEnsemble
        """
        __check_valid_connection_type__(self.__connection_type__, ['write', 'append'])
        return self.create(name=name, id=id, metadata=metadata, schema=schema, sina_type=self._ensembles_type, **kargs)

    @lock_strategies.lock_method
    def create(self, name="Unnamed Dataset", id=None,
               metadata={}, schema=None, sina_type=None, **kargs):
        """create a new (possibly named) dataset

        :param name: name for the dataset, defaults to None
        :type name: str, optional
        :param id: unique Id, defaults to None which means use uuid4()
        :type id: str, optional
        :param metadata: dictionary of attribute/value pair for the dataset, defaults to {}
        :type metadata: dict, optional
        :param schema: a KoshSchema object to validate datasets and when setting attributes
        :type schema: KoshSchema
        :param sina_type: If you want to query the store for a specific sina record type, not just a dataset
        :type sina_type: str
        :param alias_feature: Dictionary of feature aliases
        :type alias_feature: dict, opt
        :param kargs: extra keyword arguments (ignored)
        :type kargs: dict
        :raises RuntimeError: Dataset already exists
        :return: KoshDataset
        :rtype: KoshDataset
        """
        from sina.model import Record
        __check_valid_connection_type__(self.__connection_type__, ['write', 'append'])
        if "datasetId" in kargs:
            if id is None:
                warnings.warn(
                    "'datasetId' has been deprecated in favor of 'id'",
                    DeprecationWarning)
                id = kargs["datasetId"]
            else:
                raise ValueError(
                    "'datasetId' is deprecated in favor of 'id' which you already set here")

        if sina_type is None:
            sina_type = self._dataset_record_type
        if id is None:
            Id = uuid.uuid4().hex
        else:
            if id in self.__record_handler__.find_with_type(
                    sina_type, ids_only=True):
                raise RuntimeError(
                    "Dataset id {} already exists".format(id))
            Id = id

        metadata = metadata.copy()
        metadata["creator"] = self.__user_id__
        if "creation_date" in metadata:  # if statement for self.import_dataset()
            try:  # Correct format, don't do anything
                datetime.strptime(metadata["creation_date"], 'YYYY-MM-DD HH:MM:SS.microseconds')
            except ValueError:
                metadata["creation_date"] = str(datetime.fromtimestamp(time.time()))
        else:
            metadata["creation_date"] = str(datetime.fromtimestamp(time.time()))

        if "name" not in metadata:
            metadata["name"] = name
        metadata["_associated_data_"] = None
        for k in metadata:
            if k == 'alias_feature':
                metadata[k] = {'value':  kosh_pickler.dumps(metadata[k])}
            else:
                metadata[k] = {'value': metadata[k]}
        rec = Record(id=Id, type=sina_type, data=metadata, user_defined={'kosh_information': {}})
        if self.__sync__:
            _update_record(rec, self)
        else:
            self.__sync__dict__[Id] = rec
            _update_record(rec, self, self._added_unsync_mem_store)
        try:
            if sina_type == self._ensembles_type:
                out = KoshEnsemble(Id, store=self, schema=schema, record=rec)
            else:
                out = KoshDataset(Id, store=self, schema=schema, record=rec)
        except Exception as err:  # probably schema validation error
            if self.__sync__:
                _update_record(Id, self, delete=True)
            else:
                del self.__sync__dict__[Id]
                _update_record(rec, self, self._added_unsync_mem_store, delete=True)
            raise err
        return out

    @lock_strategies.lock_method
    def _find_loader(self, Id, verbose=False, requestorId=None, use_cache=True):
        """_find_loader returns a loader that can open Id

        :param Id: Id of the object to load
        :type Id: str
        :param verbose: verbose mode will show errors
        :type verbose: bool
        :param requestorId: The id of the dataset requesting data
        :type requestorId: str
        :param use_cache: do we use cached feature
        :typpe use_cache: True
        :return: Kosh loader object
        :rtype: KoshLoader
        """
        Id_original = str(Id)
        if verbose:
            print(f"Getting loader for Id: {Id}")
        if "__uri__" in Id:
            # Ok this is a pure sina file with mime_type
            Id, uri = Id.split("__uri__")
        else:
            uri = None
        if verbose:
            print("Finding loader for: {}".format(uri))
        if (Id_original, requestorId) in self._cached_loaders:
            try:
                feats = self._cached_loaders[Id_original, requestorId][0]._list_features(use_cache=use_cache)  # != []
            except Exception as err:
                feats = []
                if verbose:
                    print("Error opening {} with loader {}: {}".format(
                        uri, self._cached_loaders[Id_original, requestorId], err))
            if feats != []:
                return self._cached_loaders[Id_original, requestorId]
        record = self.get_record(Id)
        obj = self._load(Id)
        # uri not none means it is pure sina record with file and mime_type
        if (record["type"] not in self._kosh_reserved_record_types and uri is None)\
                or record["type"] in [self._ensembles_type, "__kosh_storeinfo__"]:
            # Not reserved means dataset
            return KoshSinaLoader(obj), record["type"]
        # Ok special type
        if uri is None:
            if "mime_type" in record["data"]:
                mime_type = record["data"]["mime_type"]["value"]
            else:
                mime_type = None
            mime_type_passed = mime_type
        else:  # Pure sina with file/mime_type
            mime_type = mime_type_passed = record["files"][uri]["mimetype"]
        if mime_type in self.loaders:
            for ld in self.loaders[mime_type]:
                try:
                    feats = ld(obj, mime_type=mime_type_passed,
                               uri=uri)._list_features(use_cache=use_cache, verbose=verbose)
                except Exception as err:
                    # Something happened can't list features
                    feats = []
                    if verbose:
                        print("Error opening {} with loader {}: {}".format(obj.uri, ld, err))
                if feats != []:
                    break
            self._cached_loaders[Id_original, requestorId] = ld(
                obj, mime_type=mime_type_passed, uri=uri, requestorId=requestorId), record["type"]
            return self._cached_loaders[Id_original, requestorId]
        # sometime types have subtypes (e.g 'file') let's look if we
        # understand a subtype since we can't figure it out from mime_type
        if record["type"] in self.loaders:  # ok not a generic loader let's use it
            for ld in self.loaders[record["type"]]:
                try:
                    feats = ld(obj, mime_type=mime_type_passed,
                               uri=uri, requestorId=requestorId)._list_features(use_cache=use_cache)
                except Exception:
                    # Something happened can't list features
                    feats = []
                if feats != []:
                    break
            self._cached_loaders[Id_original, requestorId] = ld(
                obj, mime_type=mime_type_passed, uri=uri, requestorId=requestorId), record["type"]
            return self._cached_loaders[Id_original, requestorId]
        return None, None

    @lock_strategies.lock_method
    def open(self, Id, loader=None, requestorId=None, *args, **kargs):
        """open loads an object in store based on its Id
        and run its open function

        :param Id: unique id of object to open. Can also be a Sina record.
        :type Id: str
        :param loader: loader to use, defaults to None which means pick for me
        :type loader: KoshLoader
        :param requestorId: The id of the dataset requesting data
        :type requestorId: str
        :return:
        """
        from sina.model import Record
        if isinstance(Id, Record):  # Sina record by itself
            Id = Id.id

        if loader is None:
            loader, _ = self._find_loader(Id, requestorId=requestorId)
        else:
            loader = loader(self._load(Id), requestorId=requestorId)
        return loader.open(*args, **kargs)

    @lock_strategies.lock_method
    def _load(self, Id):
        """_load returns an associated source based on id

        :param Id: unique id in store
        :type Id: str
        :return: loaded object
        """
        record = self.get_record(Id)
        if record["type"] == self._sources_type:
            return KoshSinaFile(Id, kosh_type=record["type"],
                                record_handler=self.__record_handler__,
                                store=self, record=record)
        else:
            return KoshSinaObject(Id, kosh_type=record["type"],
                                  record_handler=self.__record_handler__,
                                  store=self, record=record)

    @lock_strategies.lock_method
    def get(self, Id, feature, format=None, loader=None,
            transformers=[], requestorId=None, *args, **kargs):
        """get returns an associated source's data

        :param Id: Id of object to retrieve
        :type Id: str
        :param feature: feature to retrieve
        :type feature: str
        :param format: preferred format, defaults to None means pick for me
        :type format: str, optional
        :param loader: loader to use, defaults to None means pick for me
        :return: data in requested format
        :param transformers: A list of transformers to use after the data is loaded
        :type transformers: kosh.operator.KoshTransformer
        :param requestorId: The id of the dataset requesting data
        :type requestorId: str
        """
        if loader is None:
            loader, _ = self._find_loader(Id, requestorId=requestorId)
        else:
            loader = loader(self._load(Id), requestorId=requestorId)

        return loader.get(feature, format, transformers=[], *args, **kargs)

    @lock_strategies.lock_method
    def search(self, *atts, **keys):
        """
        Deprecated use find
        """
        warnings.warn("The 'search' function is deprecated and now called `find`.\n"
                      "Please update your code to use `find` as `search` might disappear in the future",
                      DeprecationWarning)
        return self.find(*atts, **keys)

    @lock_strategies.lock_method
    def find_ensembles(self, *atts, **keys):
        """Find ensembles matching some metadata in the store
        arguments are the metadata name we are looking for e.g
        find("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        :return: generator of matching ensembles in store
        :rtype: generator
        """
        return self.find(types=self._ensembles_type, *atts, **keys)

    @lock_strategies.lock_method
    def find(self, *atts, **keys):
        """Find objects matching some metadata in the store
        and its associated stores.

        Arguments are the metadata name we are looking for e.g
        find("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        "file_uri" is a reserved key that will return all records being associated
                   with the given "uri", e.g store.find(file_uri=uri)
        "types" let you search over specific sina record types only.
        "id_pool" will search based on id of Sina record or Kosh dataset. Can be a list.

        :param load_type: How the dataset is returned ('dataset' for Kosh Dataset,
            'record' for Sina Record, 'dictionary' for Dictonary).
            Used for faster load times, defaults to 'dataset'
        :type load_type: str, optional
        :return: generator of matching objects in store
        :rtype: generator
        """

        from sina.model import Record
        if 'id_pool' in keys:
            if isinstance(keys['id_pool'], str):
                ids_to_add = [keys['id_pool']]
            else:
                ids_to_add = [id for id in keys['id_pool']]
        else:
            ids_to_add = []

        keys['load_type'] = keys.get('load_type', 'dataset')
        sort_by = keys.pop("sort_by", self.session_sort_by)
        sort_by_descending = keys.pop("sort_by_descending", self.session_sort_by_descending)

        atts_to_remove = []
        for attr in atts:
            if isinstance(attr, (Record, kosh.dataset.KoshDataset)):  # Sina record or Kosh dataset by itself
                ids_to_add.append(attr.id)
                atts_to_remove.append(attr)
            elif isinstance(attr, types.GeneratorType):  # Multiple records from Sina.find() or Kosh.find()
                for at in attr:
                    ids_to_add.append(at.id)
                atts_to_remove.append(attr)

        atts = tuple([item for item in atts if item not in atts_to_remove])
        if 'id_pool' in keys or ids_to_add:  # Create key if doesn't exist
            keys['id_pool'] = [*set(ids_to_add)]

        def find_generator(*atts, **keys):
            for result in self._find(*atts, **keys):
                yield result

            searched_stores = [self.db_uri]
            if hasattr(self, "searched_stores"):
                self.searched_stores += list(searched_stores)
            else:
                self.searched_stores = list(searched_stores)
            for store in self._associated_stores_:
                if hasattr(store, "searched_stores"):
                    if store.db_uri in store.searched_stores:
                        continue
                    else:
                        store.searched_stores += list(searched_stores)
                else:
                    store.searched_stores = list(searched_stores)
                searched_stores += [store.db_uri, ]
                for result in store.find(*atts, **keys):
                    yield result
            # cleanup searched store uris
            for store in self._associated_stores_ + [self, ]:
                for id_ in list(searched_stores):
                    if id_ in store.searched_stores:
                        store.searched_stores.remove(id_)

        if sort_by is not None:
            results = list(find_generator(*atts, **keys))
            results.sort(key=lambda result: getattr(result, sort_by, -1),
                         reverse=sort_by_descending)
            for result in results:
                yield result
        else:
            yield from find_generator(*atts, **keys)

    @lock_strategies.lock_method
    def _find(self, *atts, **keys):
        """Find objects matching some metadata in the store
        arguments are the metadata name we are looking for e.g
        find("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        "file_uri" is a reserved key that will return all records being associated
                   with the given "uri", e.g store.find(file_uri=uri)
        "types" let you search over specific sina record types only.

        :param load_type: How the dataset is returned ('datset' for Kosh Dataset,
            'record' for Sina Record, 'dictionary' for Dictonary).
            Used for faster load times, defaults to 'dataset'
        :type load_type: str, optional
        :return: generator of matching objects in store
        :rtype: generator
        """
        from sina.utils import Negation, not_, exists
        mode = self.__sync__
        if mode:
            # we will not update any rec in here, turnin off sync
            # it makes things much faster
            backup = self.__sync__dict__
            self.__sync__dict__ = {}
            self.synchronous()
        sina_kargs = {}
        ids_only = keys.pop("ids_only", False)
        load_type = keys.pop("load_type", "dataset")
        # We only want to search sina for ids not records
        sina_kargs["ids_only"] = True

        if "kosh_type" in keys and "types" in keys:
            raise ValueError(
                "`kosh_type` has been replaced with `types` you cannot use both at same time")
        if "kosh_type" in keys:
            warnings.warn(
                "`kosh_type` is being deprecated in favor of `types` and will not work in a future version",
                DeprecationWarning)
            record_types = keys.pop("kosh_type")
        else:
            record_types = keys.pop("types", (None, None))  # can't use just None

        if isinstance(record_types, six.string_types):
            record_types = [record_types, ]
        if record_types not in [None, (None, None)] and not isinstance(
                record_types, (list, tuple, Negation)):
            raise ValueError("`types` must be None, str, list or sina.utils.Negation")

        if 'file_uri' in keys and 'file' in keys:
            raise ValueError(
                "`file` has been deprecated for `file_uri` but you cannot use both at same time")
        if 'file' in keys:
            file_uri = keys.pop("file")
        else:
            file_uri = keys.pop("file_uri", None)

        # Ok now let's look if the user wants to search for an id
        if "id" in keys:
            if "id_pool" in keys:
                raise ValueError("you cannot use id and id_pool together")
            warnings.warn("When searching by id use id_pool")
            sina_kargs["id_pool"] = keys["id"]
            del keys["id"]
        else:
            sina_kargs["id_pool"] = keys.pop("id_pool", None)

        sina_kargs["file_uri"] = file_uri
        sina_kargs["query_order"] = keys.pop(
            "query_order", ("data", "file_uri", "types"))

        if record_types == (None, None):
            record_types = not_(self._kosh_reserved_record_types)
            if sina_kargs["id_pool"] is not None:
                record_types = None

        sina_kargs["types"] = record_types
        # The data dict for sina
        sina_data = keys.pop("data", {})
        if not isinstance(sina_data, dict):
            keys["data"] = sina_data
            sina_data = {}
        elif len(sina_data) != 0 and (len(keys) != 0 or len(atts) != 0):
            warnings.warn(
                "It is not recommended to use the find function by mixing keys and the reserved key `data`")

        # Maybe user is trying to get an attribute data
        for att in atts:
            sina_data[att] = exists()

        sina_data.update(keys)
        sina_kargs["data"] = sina_data
        if isinstance(sina_kargs["id_pool"], six.string_types):
            sina_kargs["id_pool"] = [sina_kargs["id_pool"], ]
        # is it a blank search, e.g get me everything?
        get_all = sina_kargs.get("data", {}) == {} and \
            sina_kargs.get("file_uri", None) is None and \
            sina_kargs.get("id_pool", None) is None and \
            sina_kargs.get("types", []) == []

        if get_all:
            match = set(self.__record_handler__.get_all(ids_only=True))
        else:
            match = set(self.__record_handler__.find(**sina_kargs))

        if not self.__sync__:
            # We need to check or in memory records as well
            if get_all:
                match_mem = set(
                    self._added_unsync_mem_store.records.get_all(
                        ids_only=True))
            else:
                match_mem = set(self._added_unsync_mem_store.records.find(**sina_kargs))
            match = match.union(match_mem)

        if mode:
            # we need to restore sync mode
            self.__sync__dict__ = backup
            self.synchronous()

        for rec_id in match:
            if ids_only:
                yield rec_id
            else:
                if load_type == 'dataset':
                    try:
                        yield self.open(rec_id)
                    except Exception:
                        yield self._load(rec_id)
                elif load_type == 'record':
                    yield self._load(rec_id)
                elif load_type == 'dictionary':
                    yield self.get_record(rec_id).__dict__['raw']

    @lock_strategies.lock_method
    def check_sync_conflicts(self, keys):
        """Checks if their will be sync conflicts
        :param keys: keys of objects to syncs (id/type)
        :type keys: list
        :return: dictionary of objects ids and their failing attributes
        :rtype: dict
        """
        # First pass to make sure we have no conflict
        conflicts = {}
        for key in keys:
            try:
                db_record = self.__record_handler__.get(key)
                # db_record = self.get_record(key)
                try:
                    local_record = self.__sync__dict__[key]
                    # Dataset created locally on unsynced store do not have
                    # this attribute
                    try:
                        last_local = datetime.strptime(local_record["data"]['last_modified_date']['value'],
                                                       "%Y-%m-%d %H:%M:%S.%f").timestamp()
                    except KeyError:
                        last_local = -1
                    for att in db_record["user_defined"]['kosh_information']:
                        conflict = False
                        if att[-14:] != "_last_modified":
                            continue
                        last_db = db_record["user_defined"]['kosh_information'][att]
                        if last_db > last_local and att in local_record["user_defined"]['kosh_information']:
                            # Conflict
                            if att[-27:-14] == "___associated":
                                # ok dealing with associated data
                                uri = att[:-27]
                                # deleted locally
                                if uri not in local_record["files"]:
                                    if uri in db_record["files"]:
                                        conflict = True
                                else:
                                    if uri not in db_record["files"]:
                                        conflict = True
                                    elif db_record["files"][uri]["mimetype"] != local_record["files"][uri]["mimetype"]:
                                        conflict = True
                                if conflict:
                                    conf = {uri: (db_record["files"].get(uri, {"mimetype": "deleted"})["mimetype"],
                                                  last_db,
                                                  local_record["files"].get(uri, {"mimetype": "deleted"})[
                                        "mimetype"],
                                        local_record["user_defined"]['kosh_information'][att])}
                                    if key not in conflicts:
                                        conflicts[key] = conf
                                    else:
                                        conflicts[key].update(conf)
                                    conflicts[key]["last_check_from_db"] = last_local
                                    conflicts[key]["type"] = "associated"
                            else:
                                name = att[:-14]
                                # deleted locally
                                if name not in local_record["data"]:
                                    if name in db_record["data"]:
                                        conflict = True
                                else:
                                    if name not in db_record["data"]:
                                        conflict = True
                                    elif db_record["data"][name]["value"] != local_record["data"][name]["value"]:
                                        conflict = True
                                if conflict:
                                    conf = {name: (db_record["data"].get(name, {"value": "deleted"})["value"],
                                                   last_db,
                                                   local_record["data"].get(
                                        name, {"value": "deleted"})["value"],
                                        local_record["user_defined"]['kosh_information'][att])}
                                    if key not in conflicts:
                                        conflicts[key] = conf
                                    else:
                                        conflicts[key].update(conf)
                                    conflicts[key]["last_check_from_db"] = last_local
                                    conflicts[key]["type"] = "attribute"
                except Exception:  # ok let's see if it was a delete ones
                    local_record = self.__sync__deleted[key]
                    try:
                        last_local = datetime.strptime(local_record["data"]['last_modified_date']['value'],
                                                       "%Y-%m-%d %H:%M:%S.%f").timestamp()
                    except KeyError:
                        last_local = -1
                    for att in db_record["user_defined"]['kosh_information']:
                        conflict = False
                        if att[-14:] != "_last_modified":
                            continue
                        last_db = db_record["user_defined"]['kosh_information'][att]
                        if last_db > last_local:
                            conf = {att[:14]: (
                                "modified in db", "ds deleted here", "")}
                            if key not in conflicts:
                                conflicts[key] = conf
                            else:
                                conflicts[key].update(conf)
                            conflicts[key]["last_check_from_db"] = last_local
                            conflicts[key]["type"] = "delete"
            except BaseException:  # It's a new record no conflict
                # It could be it was deleted in store while we touched it here
                try:
                    local_record = self.__sync__dict__[key]
                    # Dataset created locally on unsynced store do not have
                    # this attribute
                    try:
                        last_local = datetime.strptime(local_record["data"]['last_modified_date']['value'],
                                                       'YYYY-MM-DD HH:MM:SS.microseconds').time()
                    except KeyError:
                        last_local = -1
                    if last_local != -1:  # yep we read it from store
                        conf = {
                            local_record["data"]["name"]["value"]: (
                                "deleted in store", "", "")}
                        conf["last_check_from_db"] = last_local
                        conf["type"] = "delete"
                        if key not in conflicts:
                            conflicts[key] = conf
                        else:
                            conflicts[key].update(conf)
                except Exception:  # deleted too so no issue
                    pass
        return conflicts

    @lock_strategies.lock_method
    def is_synchronous(self):
        """is_synchronous is store is synchronous mode

        :return: synchronous or not
        :rtype: bool
        """
        return self.__sync__

    @lock_strategies.lock_method
    def synchronous(self, mode=None):
        """Change sync mode for the store

        :param mode: The mode to True means synchronous mode, False means asynchronous, None  means switch
                     anything else is ignored and it simply returns the mode
        :type mode: bool
        :return: current synchronization mode
        :rtype: bool
        """

        if mode is None:
            self.__sync__ = not self.__sync__
        elif mode in [True, False]:
            if mode and not self.__sync__:  # Going to go to always sync on need to sync first
                self.sync()
            self.__sync__ = mode
        return self.__sync__

    @lock_strategies.lock_method
    def sync(self, keys=None):
        """Sync with db
        :param keys: keys of objects to sync (id/type)
        :type keys: list
        :return: None
        :rtype: None
        """
        if self.__sync__:
            return

        if not hasattr(self.__record_handler__, "insert"):
            raise RuntimeError("Kosh store is read_only, cannot sync with it")

        if keys is None:
            keys = list(self.__sync__dict__.keys()) + \
                list(self.__sync__deleted__.keys())
        if len(keys) == 0:
            return
        conflicts = self.check_sync_conflicts(keys)
        if len(conflicts) != 0:  # Conflicts, aborting
            msg = "Conflicts exist objects have been modified in db and locally"
            for key in conflicts:
                msg += "\nObject id:{}".format(key)
                msg += "\n\tLast read from db: {}".format(
                    conflicts[key]["last_check_from_db"])
                for k in conflicts[key]:
                    if k in ["last_check_from_db", "type"]:
                        continue
                    if conflicts[key]["type"] == "attribute":
                        st = "\n\t" + k + " modified to value '{}' at {} in db, modified locally to '{}' at {}"
                    elif conflicts[key]["type"] == "delete":
                        st = "\n\t" + k + "{} {} {}"
                    else:
                        st = "\n\tfile '" + k + \
                            "' mimetype modified to'{}' at {} in db, modified locally to '{}' at {}"
                    st = st.format(*conflicts[key][k])
                    msg += st
            raise RuntimeError(msg)
        # Ok no conflict we still need to sync
        update_records = []
        del_keys = []
        for key in keys:
            try:
                local = self.__sync__dict__[key]
            except Exception:
                # Ok it comes from the deleted datasets
                del_keys.append(key)
                continue
            try:
                db = self.__record_handler__.get(key)
                # db = self.get_record(key)
            except ValueError:
                # not in main store yet
                _update_record(local, self)
                # now we can retrieve it process for associated file not in storer yet
                db = self.__record_handler__.get(key)
                # db = self.get_record(key)
            for att in local["user_defined"]['kosh_information']:
                if att[-14:] == "_last_modified":  # We touched it
                    if att[-27:-14] == "___associated":
                        # ok it's an associated thing
                        uri = att[:-27]
                        if uri not in local["files"]:  # dissociated
                            del db["files"][uri]
                            continue
                        # Now let's see if it is in main store
                        try:
                            self.__record_handler__.get(local["files"][uri]["kosh_id"])
                            # self.get_record(local["files"][uri]["kosh_id"])
                        except ValueError:
                            # Ok it is not in the store itself
                            rec = self.get_record(local["files"][uri]["kosh_id"])
                            _update_record(rec, self)
                        if att not in db["user_defined"]['kosh_information']:  # newly associated
                            db["files"][uri] = local["files"][uri]
                            db["user_defined"]['kosh_information'][att] = \
                                local["user_defined"]['kosh_information'][att]
                        elif local["user_defined"]['kosh_information'][att] > \
                                db["user_defined"]['kosh_information'][att]:
                            # last changed locally
                            db["files"][uri] = local["files"][uri]
                            db["user_defined"]['kosh_information'][att] = \
                                local["user_defined"]['kosh_information'][att]
                    else:
                        name = att[:-14]
                        if name not in local["data"]:  # we deleted it
                            if name in db["data"]:
                                del db["data"][name]
                        elif local["user_defined"]['kosh_information'][att] > \
                                db["user_defined"]['kosh_information'][att]:
                            db["data"][name] = local["data"][name]
                            db["user_defined"]['kosh_information'][att] = \
                                local["user_defined"]['kosh_information'][att]
            if db is not None:
                update_records.append(db)
            else:  # db did not have that key and returned None (no error)
                update_records.append(local)
            del_keys.append(key)

        rels = []
        relationships = self.get_sina_store().relationships
        for id_ in update_records:
            rels += relationships.find(id_.id, None, None)
            rels += relationships.find(None, None, id_.id)
        if not self.__sync__:
            _update_record(del_keys, self, delete=True)
            _update_record(update_records, self)
            relationships.insert(rels)
        _update_record(update_records, self)
        for key in list(keys):
            try:
                _update_record(key, self, self._added_unsync_mem_store, delete=True)
            except Exception:
                pass
            try:
                del self.__sync__dict__[key]
            except Exception:
                # probably coming from del then
                del self.__sync__deleted__[key]

    @lock_strategies.lock_method
    def add_user(self, username, groups=[]):
        """add_user adds a user to the Kosh store

        :param username: username to add
        :type username: str
        :param groups: kosh specific groups to add to this user
        :type groups: list
        """
        from sina.model import Record
        __check_valid_connection_type__(self.__connection_type__, ['write', 'append'])
        existing_users = self.__record_handler__.find_with_type(
            self._users_type)
        users = [rec["data"]["username"]["value"] for rec in existing_users]
        if username not in users:
            # Create user
            uid = hashlib.md5(username.encode()).hexdigest()
            user = Record(id=uid, type=self._users_type, user_defined={'kosh_information': {}})
            user.add_data("username", username)
            _update_record(user, self)
            self.add_user_to_group(username, groups)
        else:
            raise ValueError("User {} already exists".format(username))

    @lock_strategies.lock_method
    def add_group(self, group):
        """Add a kosh specific group, cannot match existing group on unix system

        :param group: ugroup to add
        :type group: str
        """
        from sina.model import Record
        __check_valid_connection_type__(self.__connection_type__, ['write', 'append'])
        existing_groups = self.__record_handler__.find_with_type(
            self._groups_type)
        groups_names = [rec["data"]["name"]["value"]
                        for rec in existing_groups]
        if group in groups_names:
            raise ValueError("group {} already exist".format(group))

        # now get unix groups
        if not sys.platform.startswith("win"):
            unix_groups = [g[0] for g in grp.getgrall()]
        else:
            unix_groups = []
        if group in unix_groups:
            raise ValueError("{} is a unix group on this system.format(group)")

        # Create group
        uid = uuid.uuid4().hex
        group_rec = Record(id=uid, type=self._groups_type, user_defined={'kosh_information': {}})
        group_rec.add_data("name", group)
        _update_record(group_rec, self)

    @lock_strategies.lock_method
    def add_user_to_group(self, username, groups):
        """Add a user to some group(s)

        :param username: username to add
        :type username: str
        :param groups: kosh specific groups to add to this user
        :type groups: list
        """
        __check_valid_connection_type__(self.__connection_type__, ['write', 'append'])
        users_filter = self.__record_handler__.find_with_type(
            self._users_type, ids_only=True)
        names_filter = list(
            self.__record_handler__.find_with_data(
                username=username))
        inter_recs = set(users_filter).intersection(set(names_filter))
        if len(inter_recs) == 0:
            raise ValueError("User {} does not exists".format(username))
        user = self.get_record(names_filter[0])
        user_groups = user["data"].get(
            self._groups_type, {
                "value": []})["value"]

        existing_groups = self.__record_handler__.find_with_type(
            self._groups_type)
        groups_names = [rec["data"]["name"]["value"]
                        for rec in existing_groups]
        for group in groups:
            if group not in groups_names:
                warnings.warn(
                    "Group {} is not a Kosh group, skipping".format(group))
                continue
            user_groups.append(group)
        if len(user_groups) == 0:
            user.set_data("groups", None)
        else:
            user.set_data("groups", list(set(user_groups)))
        _update_record(user, self)

    @lock_strategies.lock_method
    def export_dataset(self, datasets, file=None):
        """exports a dataset

        :param datasets: dataset (or their ids) to export
        :type datasets: list or str
        :param file: optional file to dump dataset to
        :type file: None or str
        """
        if not isinstance(datasets, (list, tuple, types.GeneratorType)):
            datasets = [datasets, ]
        for dataset in datasets:
            if isinstance(dataset, six.string_types):
                return self.open(dataset).export(file)
            else:
                return dataset.export(file)

    @lock_strategies.lock_method
    def import_dataset(self, datasets, match_attributes=[
                       "name", ], merge_handler=None, merge_handler_kargs={}, skip_sina_record_sections=[],
                       ingest_funcs=None):
        """import datasets and ensembles that were exported from another store, or load them from a json file
        :param datasets: Dataset/Ensemble object exported by another store, a dataset/ensemble
                         or a json file containing these.
        :type datasets: json, csv, hdf5 file, json loaded object, kosh.KoshDataset, sina.Record, pandas.DataFrame
        :param match_attributes: parameters on a dataset to use if this it is already in the store
                                 in general we can't use 'id' since it is randomly generated at creation
                                 If the "same" dataset was created in two different stores
                                 (e.g running the same code twice but with different Kosh store)
                                 the dataset would be identical in both store but with different ids.
                                 This helps you make sure you do not end up with duplicate entries.
                                 Warning, if this parameter is too lose too many datasets will match
                                 and the import will abort, if it's too tight duplicates will not be identified.
        :type match_attributes: list of str
        :param merge_handler: If found dataset has attributes with different values from imported dataset
                                 how do we handle this? Accept values are: None, "conservative", "overwrite",
                                 "preserve", or a function.
                              The function decalartion should be:
                                        foo(store_dataset,
                                            imported_dataset_attributes_dict,
                                            section,
                                            **merge_handler_kargs)
                              Where `store_dataset` is the destination kosh dataset or its non-data dictionary section
                                    `imported_dataset_attributes_dict` is a dictionary of attributes/values
                                                                       of the dataset being imported
                                    `section` is the section of the record being updated
                                    `merge_handler_kargs` is a dict of passed for this function
                              And return a dictionary of attributes/values the target_dataset should have.
        :type merge_handler: None, str, func
        :param merge_handler_kargs: If a function is passed to merge_handler these keywords arguments
                                    will be passed in addition to this store dataset and the imported dataset.
        :type merge_handler_kargs: dict
        :param skip_sina_record_sections: When importing a sina record, skip over these sections
        :type skip_sina_record_sections: list
        :param ingest_funcs: A function or list of functions to
                             run against each Sina record before insertion.
                             We queue them up to run here. They will be run in list order.
        :type ingest_funcs: callable or list of callables
        :return: list of datasets
        :rtype: list of KoshSinaDataset
        """
        __check_valid_connection_type__(self.__connection_type__, ['write', 'append'])

        from pandas import DataFrame
        from sina.model import generate_record_from_csv, generate_record_from_pandas

        out = []
        if isinstance(datasets, DataFrame):
            datasets = generate_record_from_pandas(datasets)
        elif isinstance(datasets, str):
            if os.path.exists(datasets) and ".json" not in datasets and ".hdf5" not in datasets:
                datasets = generate_record_from_csv(datasets)

        if not isinstance(datasets, (list, tuple, types.GeneratorType)):
            return self._import_dataset(datasets, match_attributes=match_attributes,
                                        merge_handler=merge_handler,
                                        merge_handler_kargs=merge_handler_kargs,
                                        skip_sina_record_sections=skip_sina_record_sections,
                                        ingest_funcs=ingest_funcs)
        else:
            for dataset in datasets:
                out.append(self._import_dataset(dataset, match_attributes=match_attributes,
                                                merge_handler=merge_handler,
                                                merge_handler_kargs=merge_handler_kargs,
                                                skip_sina_record_sections=skip_sina_record_sections,
                                                ingest_funcs=ingest_funcs))
        return out

    @lock_strategies.lock_method
    def _import_dataset(self, datasets, match_attributes=[
            "name", ], merge_handler=None, merge_handler_kargs={}, skip_sina_record_sections=[], ingest_funcs=None):
        """import dataset that was exported from another store, or load them from a json file
        :param datasets: Dataset object exported by another store, a dataset or a json file containing the dataset
        :type datasets: json, csv, hdf5 file, json loaded object, kosh.KoshDataset, sina.Record, pandas.DataFrame
        :param match_attributes: parameters on a dataset to use if this it is already in the store
                                 in general we can't use 'id' since it is randomly generated at creation
                                 If the "same" dataset was created in two different stores
                                 (e.g running the same code twice but with different Kosh store)
                                 the dataset would be identical in both store but with different ids.
                                 This helps you make sure you do not end up with duplicate entries.
                                 Warning, if this parameter is too lose too many datasets will match
                                 and the import will abort, if it's too tight duplicates will not be identified.
        :type match_attributes: list of str
        :param merge_handler: If found dataset has attributes with different values from imported dataset
                                 how do we handle this? Accept values are: None, "conservative", "overwrite",
                                 "preserve", or a function.
                                 A function should take in foo(store_dataset, imported_dataset, **merge_handler_kargs)
        :type merge_handler: None, str, func
        :param merge_handler_kargs: If a function is passed to merge_handler these keywords arguments
                                    will be passed in addition to this store dataset and the imported dataset.
        :type merge_handler_kargs: dict
        :param skip_sina_record_sections: When importing a sina record, skip over these sections
        :type skip_sina_record_sections: list
        :param ingest_funcs: A function or list of functions to
                             run against each Sina record before insertion.
                             We queue them up to run here. They will be run in list order.
        :type ingest_funcs: callable or list of callables
        :return: list of datasets
        :rtype: list of KoshSinaDataset
        """
        from sina.model import Relationship, generate_record_from_json, Record
        from sina.utils import load_document, load_document_hdf5

        if isinstance(datasets, Record):
            json_data = datasets.to_json()
            if isinstance(json_data, bytes):
                json_str = json_data.decode('utf-8')
            else:
                json_str = json_data
            json_str = re.sub(r'\bNaN\b', 'null', json_str, flags=re.IGNORECASE)
            parsed = orjson.loads(json_str)
            datasets = {"records": [parsed]}

        if isinstance(datasets, str):
            # First let's try using Sina's load
            try:
                try:
                    recs, relationships_in = load_document_hdf5(datasets)
                except:  # noqae722
                    recs, relationships_in = load_document(datasets)
                records_in = [rec.raw for rec in recs]
                from_file = False
            except Exception:
                with open(datasets) as f:
                    from_file = orjson.loads(f.read())
                    if isinstance(from_file, list):
                        records_in = []
                        relationships_in = []
                        for entry in from_file:
                            records_in.append(entry.get("records", []))
                            relationships_in.append(entry.get("relationships", []))
                    else:
                        records_in = from_file.get("records", [])
                        relationships_in = from_file.get("relationships", [])
        elif isinstance(datasets, dict):
            from_file = datasets
            records_in = from_file["records"]
            relationships_in = from_file.get("relationships", [])
        elif isinstance(datasets, (KoshDataset, KoshEnsemble)):
            from_file = datasets.export()
            records_in = from_file["records"]
            relationships_in = from_file.get("relationships", [])
        else:
            raise ValueError(
                "`datasets` must be a Kosh importable object or a file or dict containing json-ized datasets")

        if ingest_funcs is not None:
            temp_store = connect(None)
            temp_store.get_sina_records().insert(
                [generate_record_from_json(record) for record in records_in])
            temp_datasets = list(temp_store.find())
            if isinstance(ingest_funcs, (list, tuple)):
                for ingest_func in ingest_funcs:
                    temp_datasets = [ingest_func(ds) for ds in temp_datasets]
            else:
                temp_datasets = [ingest_func(ds) for ds in temp_datasets]
            records_in = [ds.export()["records"][0] for ds in temp_datasets]
        # setup merge handler
        ok_merge_handler_values = [
            None, "conservative", "preserve", "overwrite"]
        if merge_handler in ok_merge_handler_values:
            merge_handler_kargs = {"handling_method": merge_handler}
            merge_handler = merge_datasets_handler
        elif not (isfunction(merge_handler) or ismethod(merge_handler)):
            raise ValueError(
                "'merge_handler' must be one {} or a function/method".format(ok_merge_handler_values))

        matches = []
        remapped = {}
        for record in records_in:
            if "id" not in record:
                if "local_id" in record:
                    record["id"] = record["local_id"]
                else:  # Cannot import a rec w/o id
                    warnings.warn(f"Skipped record w/o id or local_id: {record}")
                    continue
            if "name" not in record["data"]:
                record["data"]["name"] = {"value": "Unnamed Dataset"}
            if "creator" not in record["data"]:
                record["data"]["creator"] = {"value": self.__user_id__}
            if "creation_date" not in record["data"]:
                record["data"]["creation_date"] = {"value": str(datetime.fromtimestamp(time.time()))}
            if 'user_defined' not in record.keys():
                record["user_defined"] = {}
            record["user_defined"]['kosh_information'] = {}
            for section in skip_sina_record_sections:
                record[section] = {}
            if "data" in record:
                data = record["data"]
            else:
                data = {}
            if from_file and record["type"] == from_file.get("sources_type", "file"):
                is_source = True
            else:
                is_source = False
            # Not 100% data.keys is guaranteed to come back the same twice in a
            # row
            keys = sorted(data.keys())
            atts = dict(zip(keys, [data[x]["value"] for x in keys]))
            min_ver = from_file.get("minimum_kosh_version", (0, 0, 0)) if from_file else None
            if min_ver is not None and kosh.version(comparable=True) < min_ver:
                raise ValueError("Cannot import dataset it requires min kosh version of {}, we are at: {}".format(
                    min_ver, kosh.version(comparable=True)))

            if not is_source:
                # Ok now we need to see if dataset already exist?
                match_dict = {}
                for attribute in match_attributes:
                    if attribute in atts:
                        match_dict[attribute] = atts[attribute]
                    elif attribute == "id":
                        match_dict["id_pool"] = record["id"]

                matching = list(self.find(**match_dict))
                if len(matching) > 1:
                    raise ValueError("dataset criteria: {} matches multiple ({}) "
                                     "datasets in store {}, try changing 'match_attributes' when calling"
                                     " this function".format(
                                         match_dict, len(matching), self.db_uri))
                elif len(matching) == 1:
                    # All right we do have a possible conflict here
                    match = matching[0]
                    merged_attributes = merge_handler(
                        match, atts, "data", **merge_handler_kargs)
                    # Ok at this point no conflict!
                    match.update(merged_attributes)
                    match_rec = match.get_record()
                    remapped[record["id"]] = match_rec.id
                else:  # Non existent dataset
                    try:
                        self.get_record(record["id"])
                        # Ok this record already exists
                        # and we need a new unique one
                        record["id"] = uuid.uuid4().hex
                    except (KeyError, ValueError):
                        # Does not exists, let's keep the id
                        pass
                    match_rec = record
            else:  # ok it is a source
                # Let's find the source rec that match this uri
                current_sources = list(
                    self.find(
                        types=[self._sources_type, ],
                        uri=data["uri"]["value"],
                        ids_only=True))
                if len(current_sources) > 0:
                    match = self._load(current_sources[0])
                    if match.id != record["id"]:
                        # Darn this store already associated this uri
                        if match.mime_type != data["mime_type"]["value"]:
                            raise ValueError("trying to import an associated source {} with mime_type {} but "  # noqa
                                             "this store already associated"  # noqa
                                             " this source with mime_type {}".format(data["uri"]["value"],
                                                                                     data["mime_type"]["value"],
                                                                                     match.mime_type))
                    match_rec = match.get_record()
                else:
                    match_rec = record
            # update the record
            # But first make sure it is a record :)
            if isinstance(match_rec, dict):
                if 'id' not in match_rec:
                    match_rec['id'] = uuid.uuid4().hex
                if from_file:
                    match_rec = generate_record_from_json(match_rec)
                else:
                    for rec in recs:
                        if rec["id"] == record["id"]:
                            match_rec = rec
                            break

            # User defined and files are preserved?
            for section in ["user_defined", "files", "library_data"]:
                if section in record:
                    if match_rec.raw[section] != record[section] and record[section] != {
                    }:
                        if merge_handler != merge_datasets_handler:
                            match_rec.raw[section].update(
                                merge_handler(match_rec, record[section],
                                              section, **merge_handler_kargs))
                        elif merge_handler_kargs["handling_method"] == "conservative":
                            raise RuntimeError(
                                "{} section do not match aborting under conservative merge option"
                                "\nImport: {}\nInstore: {}".format(section, record[section], match_rec.raw[section]))
                        elif merge_handler_kargs["handling_method"] == "overwrite":
                            match_rec.raw[section].update(record[section])
                        else:  # preserve
                            pass
            # Curves are preserved
            if "curve_sets" in record:
                for curve_set in record["curve_sets"]:
                    if curve_set not in match_rec.raw["curve_sets"]:
                        match_rec.raw["curve_sets"][curve_set] = record["curve_sets"][curve_set]
                    else:
                        if merge_handler != merge_datasets_handler:
                            raise RuntimeError(
                                "We do not know how to merge curves with custom merge handler")
                        if merge_handler_kargs["handling_method"] == "conservative":
                            if match_rec.raw["curve_sets"][curve_set] != record["curve_sets"][curve_set]:

                                raise RuntimeError(
                                    "curveset {} do not match, `conservative` method used, aborting".format(curve_set))
                        elif merge_handler_kargs["handling_method"] == "overwrite":
                            match_rec.raw["curve_sets"][curve_set]["independent"].update(
                                record["curve_sets"][curve_set]["independent"])
                            match_rec.raw["curve_sets"][curve_set]["dependent"].update(
                                record["curve_sets"][curve_set]["dependent"])
                        else:  # preserve
                            pass
            _update_record(match_rec, self)
            matches.append(match_rec["id"])

        for relationship in relationships_in:
            try:
                rel = Relationship(subject_id=getattr(relationship, 'subject', relationship.subject_id),
                                   predicate=relationship.predicate,
                                   object_id=getattr(relationship, 'object', relationship.object_id))
                self.get_sina_store().relationships.insert(rel)
            except Exception:  # sqlalchemy.exc.IntegrityError
                pass

            # Ensembles
            if getattr(relationship, 'predicate', '') == 'is a member of ensemble':
                try:
                    self.create_ensemble(id=relationship.object_id)
                except Exception:  # ensemble already in store
                    pass

                try:
                    e = list(self.find(id=relationship.object_id))[0]
                    ds = list(self.find(id=relationship.subject_id))[0]
                    ds.join_ensemble(e)
                except Exception:  # dataset already in ensemble
                    pass

        # We need to make sure any merged (remapped) dataset is still properly
        # associated
        for id_ in matches:
            rec = self.get_record(id_)
            if rec["type"] == self._sources_type == from_file.get(
                    "sources_type", "file"):
                associated = rec["data"]["associated"]["value"]
                altered = False
                for rem in remapped:
                    if rem in associated:
                        index = associated.index(rem)
                        associated[index] = remapped[rem]
                        d = self.open(remapped[rem])
                        d.associate(
                            rec["data"]["uri"]["value"],
                            rec["data"]["mime_type"]["value"])
                        altered = True
                if altered:
                    _update_record(rec, self)

        return [self.open(x) for x in matches]

    @lock_strategies.lock_method
    def reassociate(self, target, source=None, absolute_path=True):
        """This function allows to re-associate data whose uri might have changed

        The source can be the original uri or sha and target is the new uri to use.
        :param target: New uri
        :type target: str
        :param source: uri or sha (long or short of reassociate) to reassociate
                       with target, if None then the short uri from target will be used
        :type source: str or None
        :param absolute_path: if file exists should we store its absolute_path
        :type absolute_path: bool
        :return: None
        :rtype: None
        """
        __check_valid_connection_type__(self.__connection_type__, ['write'])
        # First let's convert to abs path if necessary
        if absolute_path:
            if os.path.exists(target):
                target = os.path.abspath(target)
            if source is not None and not re.search("[0-9a-f]{64}", source):  # full path doesn't apply to sha
                source = os.path.abspath(source)

        if os.path.isdir(target):
            target = os.path.join(target, "")
            cmd = "rsync -v --dry-run -r" + " " + target + " ./"
            p, o, e = process_cmd(cmd, use_shell=True)
            rsync_dryrun_out_lines = o.decode().split("\n")
            index = [idx for idx, s in enumerate(rsync_dryrun_out_lines) if 'sent ' in s][0]
            targets_in_dir = rsync_dryrun_out_lines[1:index-1]
            targets = [os.path.join(target, target_in_dir) for target_in_dir in targets_in_dir]
        else:
            targets = [target]

        for target in targets:
            # Now, did we pass a source for uri to replace?
            if source is None:
                source = compute_fast_sha(target)

            # Ok now let's get all associated uri that match
            # Fist assuming it's a fast_sha search all "kosh files" that match this
            matches = list(
                self.find(
                    types=[
                        self._sources_type,
                    ],
                    fast_sha=source,
                    ids_only=True))
            # Now it could be simply a uri
            matches += list(
                self.find(
                    types=[
                        self._sources_type,
                    ],
                    uri=source,
                    ids_only=True))
            # And it's quite possible it's a long_sha too
            matches += list(self.find(types=[self._sources_type, ],
                                      long_sha=source, ids_only=True))

            # And now let's do the work
            for match_id in matches:
                try:
                    match = self._load(match_id)
                    for associated_id in match.associated:
                        associated = self.open(associated_id)
                        associated_record = associated.get_record()
                        raw_associated_record = associated_record.raw
                        raw_associated_record["files"][target] = raw_associated_record["files"][match.uri]
                        del raw_associated_record["files"][match.uri]
                        if self.sync:
                            associated._update_record(associated_record)
                        else:
                            associated._update_record(associated_record, self._added_unsync_mem_store)
                    match.uri = target
                except Exception:
                    pass

    @lock_strategies.lock_method
    def cleanup_files(self, dry_run=False, interactive=False, clean_fastsha=False,
                      **dataset_search_keys):
        """Cleanup the store from references to dead files
        Also updates the fast_shas if necessary
        You can filter associated objects for each dataset by passing key=values
        e.g mime_type=hdf5 will only dissociate non-existing files associated with mime_type hdf5
        some_att=some_val will only dissociate non-existing files associated and having the attribute
        'some_att' with value of 'some_val'
        returns list of uris to be removed.
        :param dry_run: Only does a dry_run
        :type dry_run: bool
        :param interactive: interactive mode, ask before dissociating
        :type interactive: bool
        :param clean_fastsha: Do we want to update fast_sha if it changed?
        :type clean_fastsha: bool
        :returns: list of uris (to be) removed.
        :rtype: list
        """
        __check_valid_connection_type__(self.__connection_type__, ['write'])
        missings = []
        datasets = self.find()
        for dataset in datasets:
            missings += dataset.cleanup_files(dry_run=dry_run,
                                              interactive=interactive,
                                              clean_fastsha=clean_fastsha,
                                              **dataset_search_keys)
        return missings

    @lock_strategies.lock_method
    def check_integrity(self):
        """Runs a sanity check on the store:
        1- Are associated files reachable?
        2- Did fast_shas change since file was associated
        """
        return self.cleanup_files(dry_run=True, clean_fastsha=True)

    @lock_strategies.lock_method
    def associate(self, store, reciprocal=False):
        """Associate another store

        All associated stores will be used for queries purposes.

        WARNING: While associating stores will make them look like one big store,
                 ensembles' members MUST belong to the same store as the ensemble.

        :param store: The store to associate
        :type store: KoshStore

        :param reciprocal: By default, this is a one way relationship.
                           The associated store will NOT be aware of
                           this association, turning this on create
                           the association in both stores.
        :type reciprocal: bool
        """
        __check_valid_connection_type__(self.__connection_type__, ['write', 'append'])
        if not isinstance(store, KoshStore):
            raise TypeError("store must be a KoshStore or path to one")

        sina_recs = self.get_sina_records()
        store_info = list(sina_recs.find_with_type("__kosh_storeinfo__"))[0]
        if "associated_stores" not in store_info["data"]:
            store_info.add_data("associated_stores", [])
        stores = store_info["data"]["associated_stores"]["value"]
        if store.db_uri not in stores:
            stores.append(store.db_uri)
            store_info["data"]["associated_stores"]["value"] = stores
            _update_record(store_info, self)
            self._associated_stores_.append(store)
        if reciprocal:
            store.associate(self)

    @lock_strategies.lock_method
    def dissociate(self, store, reciprocal=False):
        """Dissociate another store

        :param store: The store to associate
        :type store: KoshStore or six.string_types

        :param reciprocal: By default, this is a one way relationship.
                           The disssociated store will NOT be aware of
                           this action, turning this on create
                           the dissociation in both stores.
        :type reciprocal: bool
        """
        __check_valid_connection_type__(self.__connection_type__, ['write'])
        if not isinstance(store, (six.string_types, KoshStore)):
            raise TypeError("store must be a KoshStore or path to one")

        sina_recs = self.get_sina_records()
        store_info = list(sina_recs.find_with_type("__kosh_storeinfo__"))[0]
        if "associated_stores" not in store_info["data"]:
            warnings.warn("No store is associated with this store: {}".format(self.db_uri))
            return
        # refresh value
        stores = store_info["data"]["associated_stores"]["value"]

        if isinstance(store, six.string_types):
            try:
                store_path = store
                store = self.get_associated_store(store_path)
            except Exception:
                raise ValueError("Could not open store at: {}".format(store_path))

        if store.db_uri in stores:
            stores.remove(store.db_uri)
            store_info["data"]["associated_stores"]["value"] = stores
            _update_record(store_info, self)
            self._associated_stores_.remove(store)
        else:
            warnings.warn("store {} does not seem to be associated with this store ({})".format(
                store.db_uri, self.db_uri))

        if reciprocal:
            store.dissociate(self)

    @lock_strategies.lock_method
    def get_associated_store(self, uri):
        """Returns the associated store based on its uri.

        :param uri: uri to the desired store
        :type uri: six.string_types
        :returns: Associated kosh store
        :rtype: KoshStore
        """

        if not isinstance(uri, six.string_types):
            raise TypeError("uri must be string")

        for store in self._associated_stores_:
            if store.db_uri == uri:
                return store
        raise ValueError(
            "{} store does not seem to be associated with this store: {}".format(uri, store.db_uri))

    @lock_strategies.lock_method
    def get_associated_stores(self, uris=True):
        """Return the list of associated stores
        :param uris: Return the list of uri pointing to the store if True,
                     or the actual stores otherwise.
        :type uris: bool
        :returns: generator to stores
        :rtype: generator
        """
        for store in self._associated_stores_:
            if uris:
                yield store.db_uri
            else:
                yield store

    @lock_strategies.lock_method
    def _cli_list_creator(self, arg, var, cmmd, path=""):
        """Creates a list of arg and var pairs for the cmmd passed to kosh_command.py

        :param arg: The argparse argument to use
        :type arg: str
        :param var: The variable that goes along with the argparse argument
        :type var: str
        :param cmmd: The command list to append to
        :type cmmd: list
        :param path: Path that will be combined with var, defaults to ""
        :type path: str, optional
        :return: The appended command list
        :rtype: list
        """

        if isinstance(var, str):
            var = f"{arg} " + os.path.join(path, var)
        elif isinstance(var, list):
            if len(var) == 1:
                var = f"{arg} " + os.path.join(path, var[0])
            else:
                for i, v in enumerate(var):
                    var[i] = os.path.join(path, v)
                var = f"{arg} " + f" {arg} ".join(var)
        var = var.split()

        cmmd = cmmd + var

        return cmmd

    @lock_strategies.lock_method
    def _mv_cp(self, src, dst, mv_cp,
               stores, destination_stores, dataset_record_type,
               dataset_matching_attributes, version, merge_strategy, mk_dirs):
        """Creates the cmmd for mv and cp passed to kosh_command.py

        :param src: The source of files or directories to mv or cp
        :type src: Union[str, list]
        :param dst: The destination of files or directories to mv or cp
        :type dst: str
        :param mv_cp: Move or copy files or directories
        :type mv_cp: str
        :param stores: Kosh stores to associate the mv or cp
        :type stores: Union[kosh.dataset.KoshDataset, list]
        :param destination_stores: Kosh stores to associate the mv or cp
        :type destination_stores: Union[kosh.dataset.KoshDataset, list]
        :param dataset_record_type: Type used by sina db that Kosh will recognize as dataset
        :type dataset_record_type: str
        :param dataset_matching_attributes: List of attributes used to identify if two datasets are identical
        :type dataset_matching_attributes: list
        :param version: Print version and exit
        :type version: bool
        :param merge_strategy: When importing dataset, how do we handle conflict
        :type merge_strategy: str
        :param mk_dirs: Make destination directories if they don't exist
        :type mk_dirs: bool
        """

        # --stores
        cmmd = ["--stores",  self.db_uri]

        if stores:
            if isinstance(stores, list):
                for i, store in enumerate(stores):
                    stores[i] = store.db_uri
            else:
                stores = stores.db_uri
            cmmd = self._cli_list_creator("--stores", stores, cmmd, os.getcwd())

        # --destination_stores
        if destination_stores:
            if isinstance(destination_stores, list):
                for i, destination_store in enumerate(destination_stores):
                    destination_stores[i] = destination_store.db_uri
            else:
                destination_stores = destination_stores.db_uri
            cmmd = self._cli_list_creator("--destination_stores", destination_stores, cmmd, os.getcwd())

        # --sources
        cmmd = self._cli_list_creator("--sources", src, cmmd)

        # --dataset_record_type
        cmmd.extend(["--dataset_record_type", dataset_record_type])

        # --dataset_matching_attributes
        cmmd.extend(["--dataset_matching_attributes", f"{dataset_matching_attributes}"])

        # --destination
        cmmd.extend(["--destination", dst])

        # --version
        if version:
            cmmd.extend(["--version"])

        # --merge_strategy
        cmmd.extend(["--merge_strategy", merge_strategy])

        # --mk_dirs
        if mk_dirs:
            cmmd.extend(["--mk_dirs"])

        KoshCmd._mv_cp_(self, mv_cp, store_args=cmmd)

    @lock_strategies.lock_method
    def mv(self, src, dst, stores=[],
           destination_stores=[], dataset_record_type="dataset", dataset_matching_attributes=['name', ],
           version=False, merge_strategy="conservative", mk_dirs=False):
        """Moves files or directories

        :param src: The source of files or directories to mv or cp
        :type src: Union[str, list]
        :param dst: The destination of files or directories to mv or cp
        :type dst: str
        :param stores: Kosh stores to associate the mv or cp, defaults to []
        :type stores: Union[kosh.dataset.KoshDataset, list], optional
        :param destination_stores: Kosh stores to associate the mv or cp, defaults to []
        :type destination_stores: Union[kosh.dataset.KoshDataset, list], optional
        :param dataset_record_type: Type used by sina db that Kosh will recognize as dataset, defaults to "dataset"
        :type dataset_record_type: str, optional
        :param dataset_matching_attributes: List of attributes used to identify if two datasets are identical,
            defaults to ["name", ]
        :type dataset_matching_attributes: list, optional
        :param version: Print version and exit, defaults to False
        :type version: bool, optional
        :param merge_strategy: When importing dataset, how do we handle conflict, defaults to "conservative"
        :type merge_strategy: str, optional
        :param mk_dirs: Make destination directories if they don't exist
        :type mk_dirs: bool, optional
        """
        __check_valid_connection_type__(self.__connection_type__, ['write'])
        self._mv_cp(src, dst, "mv", stores, destination_stores, dataset_record_type,
                    dataset_matching_attributes, version, merge_strategy, mk_dirs)

    @lock_strategies.lock_method
    def cp(self, src, dst, stores=[],
           destination_stores=[], dataset_record_type="dataset", dataset_matching_attributes=['name', ],
           version=False, merge_strategy="conservative", mk_dirs=False):
        """Copies files or directories

        :param src: The source of files or directories to mv or cp
        :type src: Union[str, list]
        :param dst: The destination of files or directories to mv or cp
        :type dst: str
        :param stores: Kosh stores to associate the mv or cp, defaults to []
        :type stores: Union[kosh.dataset.KoshDataset, list], optional
        :param destination_stores: Kosh stores to associate the mv or cp, defaults to []
        :type destination_stores: Union[kosh.dataset.KoshDataset, list], optional
        :param dataset_record_type: Type used by sina db that Kosh will recognize as dataset, defaults to "dataset"
        :type dataset_record_type: str, optional
        :param dataset_matching_attributes: List of attributes used to identify if two datasets are identical,
            defaults to ["name", ]
        :type dataset_matching_attributes: list, optional
        :param version: Print version and exit, defaults to False
        :type version: bool, optional
        :param merge_strategy: When importing dataset, how do we handle conflict, defaults to "conservative"
        :type merge_strategy: str, optional
        :param mk_dirs: Make destination directories if they don't exist
        :type mk_dirs: bool, optional
        """
        __check_valid_connection_type__(self.__connection_type__, ['write', 'append'])
        self._mv_cp(src, dst, "cp", stores, destination_stores, dataset_record_type,
                    dataset_matching_attributes, version, merge_strategy, mk_dirs)

    @lock_strategies.lock_method
    def tar(self, tar_file, tar_opts, src="", tar_type="tar",
            stores=[], dataset_record_type="dataset", no_absolute_path=False,
            dataset_matching_attributes=["name", ], merge_strategy="conservative"):
        """Creates or extracts a tar file

        :param tar_file: The name of the tar file
        :type tar_file: str
        :param tar_opts: Extra arguments such as -c to create and -x to extract
        :type tar_opts: str
        :param src: List of files or directories to tar
        :type src: list, optional
        :param tar_type: Type of tar file including htar, defaults to "tar"
        :type tar_type: str, optional
        :param stores: Kosh store(s) to use, defaults to []
        :type stores: list, optional
        :param dataset_record_type: Record type used by Kosh when adding
            datasets to Sina database, defaults to "dataset"
        :type dataset_record_type: str, optional
        :param no_absolute_path: Do not use absolute path when searching stores, defaults to False
        :type no_absolute_path: bool, optional
        :param dataset_matching_attributes: List of attributes used to identify if two datasets
            are identical, defaults to ["name", ]
        :type dataset_matching_attributes: list, optional
        :param merge_strategy: When importing dataset, how do we handle conflict, defaults to "conservative"
        :type merge_strategy: str, optional
        """

        # Options includes src
        opts = tar_opts.split()
        if isinstance(src, str):
            src = [src]
        opts.extend(src)

        # --stores
        cmmd = ["--stores",  self.db_uri]

        if stores:
            if isinstance(stores, list):
                for i, store in enumerate(stores):
                    stores[i] = store.db_uri
            else:
                stores = stores.db_uri
            cmmd = self._cli_list_creator("--stores", stores, cmmd, os.getcwd())

        # --dataset_record_type
        cmmd.extend(["--dataset_record_type", dataset_record_type])

        # --file
        cmmd.extend(["--file",  tar_file])

        # --no_absolute_path
        if no_absolute_path:
            cmmd.extend(["--no_absolute_path"])

        # --dataset_matching_attributes
        cmmd.extend(["--dataset_matching_attributes", f"{dataset_matching_attributes}"])

        # --merge_strategy
        cmmd.extend(["--merge_strategy", merge_strategy])

        KoshCmd._tar(self, tar_type, store_args=cmmd, opts=opts)

    @lock_strategies.lock_method
    def to_dataframe(self, data_columns=[], typed_columns=None, *atts, **keys):
        """Return the find object as a Pandas DataFrame.

        Pass in the same arguments and keyword arguments as the find method.

        Arguments are the metadata name we are looking for e.g
        find("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        "file_uri" is a reserved key that will return all records being associated
                   with the given "uri", e.g store.find(file_uri=uri)
        "types" let you search over specific sina record types only.
        "id_pool" will search based on id of Sina record or Kosh dataset. Can be a list.

        :param data_columns: Columns to extract. By default this will include ['id', 'name', 'creator',
                                                                               'creation_date', 'last_modified_date'].
                             If nothing is passed, will return all data.
        :type data_columns: Union(str, list), optional
        :param typed_columns: Dictionary of column types and replacement values.
            If omitted, MySQL defaults are returned.

            Supported types:
                [
                    "Int8", "Int16", "Int32", "Int64",
                    "UInt8", "UInt16", "UInt32", "UInt64",
                    "float16", "float32", "float64",
                    "Float32", "Float64",
                    "bool", "boolean", "string", "object",
                    "category", "datetime64[ns]", "timedelta64[ns]",
                ]

            Example:
                {
                    "my_int": {
                        "type": "Int64",
                        "replace": {None: 42},
                    },
                    "my_float": {
                        "type": "float64",
                        "replace": {None: np.nan, 999: -1},
                    },
                    "my_boolean": {
                        "type": "boolean",
                        "replace": {None: False},
                    },
                    "my_string": {
                        "type": "string",
                        "replace": {"This is empty": "Nothing Here"},
                    },
                }
        :type typed_columns: dict, optional
        :return: Pandas DataFrame
        :rtype: Pandas DataFrame
        """
        import pandas as pd
        if isinstance(data_columns, str):
            data_columns = [data_columns]

        keys['load_type'] = 'dictionary'
        keys['ids_only'] = False
        datasets = list(self.find(*atts, **keys))

        attr_dict = {}
        total_datasets = len(datasets)

        # Always have these by default
        defaults = ['id', 'name', 'creator', 'creation_date', 'last_modified_date']

        # Acquire all data if `data_columns` was not passed
        if not data_columns:
            unique_keys = []
            for i, dataset in enumerate(datasets):
                unique_keys.extend(list(dataset['data'].keys()))

            data_columns = sorted(set(unique_keys))

        data_columns = defaults + data_columns  # Want defaults in front
        data_columns = [dc for dc in data_columns if "_ENSEMBLE_TAG_" not in dc]  # Remove ensemble tags
        attr_dict = {d: [pd.NA] * total_datasets for d in data_columns}

        for i, dataset in enumerate(datasets):
            for column in data_columns:
                if column == "id":
                    attr_dict[column][i] = dataset['id']
                else:
                    attr_dict[column][i] = dataset['data'].get(column, {}).get('value', pd.NA)

        df = pd.DataFrame(attr_dict)

        if isinstance(typed_columns, dict):
            for key, config in typed_columns.items():
                series = df[key]

                if "replace" in config:
                    series = series.replace(config["replace"]).infer_objects(copy=False)

                df[key] = series.astype(config["type"])

        return df
