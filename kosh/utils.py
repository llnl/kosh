from __future__ import absolute_import
import collections
import importlib.metadata
import os
import time
from datetime import datetime
import kosh
import hashlib
import random
import warnings
import pickle
from .wrapper import KoshScriptWrapper # noqa
from . import lock_strategies
try:
    import orjson
except ImportError:
    import json as orjson  # noqa


class KoshPickler(object):
    def __init__(self, encode_types=["windows-1252", "latin1"]):
        self.encode_types = encode_types

    def dumps(self, data):
        success = False
        for encode_type in self.encode_types:
            try:
                encoded = pickle.dumps(data).decode(encode_type)
                self.loads(encoded)
                success = True
                break
            except Exception:
                pass
        if success:
            return encoded
        else:
            raise RuntimeError("Could not pickle")

    def loads(self, data):
        success = False
        for encode_type in self.encode_types:
            try:
                loaded = pickle.loads(data.encode(encode_type))
                success = True
                break
            except Exception:
                pass
        if success:
            return loaded
        else:
            raise RuntimeError("Could not load pickled")


def find_curveset_and_curve_name(name, rec):
    """Given a curveset or curveset+curve name,
    returns all matching curve_sets and curves combinations
    Assumes curveset and curve are separated by a /
    curve_sets that exactly match the name return (name, None)
    :param name: Name to parse
    :type name: str
    :param rec: sina record where to look for curves
    :type rec: sina record
    :return: All possible combinations of (curveset,curve) that match name
    :rtype: tuple of tuples
    """
    sp = name.split("/")
    possibilities = ()
    for i in range(len(sp)+1):
        curve_set = "/".join(sp[:i])
        if curve_set in rec["curve_sets"]:
            cs = rec["curve_sets"][curve_set]
            curve_name = "/".join(sp[i:])
            if curve_name in cs["dependent"] or curve_name in cs["independent"]:
                possibilities += ((curve_set, curve_name), )
            if curve_set == name:  # ok we asked for the curve_set
                possibilities += ((curve_set, None),)
    return possibilities


def merge_datasets_handler(target_dataset, imported_dataset, section="data", **kargs):
    """When importing a dataset, checks if the imported dataset has
    attributes that match the one in the dataset already in this store.
    If attributes values conflict then we use 'handling_method to resolve the conflict

    The store_dataset is not updated here, we return a list of attributes/values pairs resolving the conflict

    :param target_dataset: The dataset that will received the merge
    :type target_dataset: kosh.KoshDataset
    :param imported_dataset: The dataset we are trying to merge into target_dataset or its attributes/values dictionary
    :type imported_dataset: kosh.KoshDataset or dict
    :param section: The section being updated (data, user_defined, curves, etc...)
    :type section: str
    :param handling_method: How do we handle conflicts?
                            None, "conservative": Error exit
                            "preserve": Keep value from target_dataset
                            "overwrite": Use value from imported dataset
    :returns: Dictionary of attribute/value that the target_dataset should have
    :rtype: dict
    """
    if section != "data":
        raise ValueError("This handler cannot handle non 'data' section")

    handling_method = kargs.pop("handling_method", None)

    target_dict = target_dataset.list_attributes(dictionary=True)

    if not isinstance(imported_dataset, dict):
        imported_dataset = imported_dataset.list_attributes(dictionary=True)

    # We cannot set _associatated_data_ anyway and if it comes last (py2) it
    # prevents updating the db
    imported_dataset.pop("_associated_data_", None)
    # Creator is a pain looks like it comes with id at times and actual user name at others
    # TODO: take a closer look in a separate MR
    if "creator" in target_dict:
        imported_dataset.pop("creator", None)

    for attribute, value in imported_dataset.items():
        if attribute in target_dict and attribute not in ['creation_date',
                                                          'last_modified_date']:
            if target_dict[attribute] != value:
                if handling_method in [None, "conservative"]:
                    msg = "Trying to import dataset with attribute '{}'".format(
                        attribute)
                    msg += " value : {}. ".format(value)
                    msg += "But value for this attribute in target is '{}'".format(
                        target_dict[attribute])
                    raise ValueError(msg)
                elif handling_method == "overwrite":
                    # Do we want a warning here?
                    # handling says use new value
                    target_dict[attribute] = value
                elif handling_method == "preserve":
                    # Do we want a warning here?
                    # We preserve so not changing the target value
                    pass
                else:
                    raise ValueError(
                        "Unknown 'handling_method': {}".format(handling_method))
        else:
            # New attribute let's add it
            target_dict[attribute] = value
    return target_dict


def gen_labels(G):
    """Generates labels to draw on networkx plots of a graph
    :param G: Network to generate labels from
    :type G: networkx.DiGraph (OrderedDiGraph on older version of networkx)
    :returns: labels for this graph
    :rtype: dict
    """
    labels = {}
    cont = True
    nodes = list(G.nodes())
    N = len(nodes)
    while cont:
        for node in nodes:
            if G.nodes[node].get("depth", None) is not None:
                continue
            pre = list(G.predecessors(node))
            suc = list(G.successors(node))
            if len(pre) == 0:
                G.nodes[node]["depth"] = 0
            else:
                for pnode in pre:
                    if G.nodes[pnode].get("depth", None) is not None:
                        G.nodes[node]["depth"] = G.nodes[pnode]["depth"] + 1
            if len(suc) == 0:
                G.nodes[node]["depth"] = -1
        total = 0
        for node in nodes:
            if G.nodes[node].get("depth", None) is not None:
                total += 1
        if total == N:
            cont = False
    for node in nodes:
        depth = G.nodes[node]["depth"]
        if depth == 0:
            depth = "start"
        elif depth == -1:
            depth = "end"
        try:
            name = node[1].__name__
        except BaseException:
            name = str(node[1].__class__).split(".")[-1].split("'")[0]
            if isinstance(node[1], kosh.loaders.core.KoshLoader):
                name = "{}({})".format(name, node[1].feature)
        labels[node] = "{}/{}/{}".format(depth, name, node[0])
    return labels


def draw_execution_graph(G,
                         output_format=None,
                         png_name="kosh_execution_graph.png",
                         clear=True,
                         layout=None):
    """Draws the graph and if provided an output format, draws the shortest path to it
    :param G: networkx graph or KoshExecutionGraph
    :type G: networkx.Graph
    :param output_format: draw shortest path to this format
    :type output_format: str or None
    :param png_name: name of png file to output the graph to
    :type png_name: str
    :param clear: clear matplotlib figure after saving
    :type clear: bool
    :param layout: A dictionary with nodes as keys and positions as values.
                   If not specified a planar layout positioning will be computed.
                   See networkx.drawing.layout for functions that compute node positions.
    :type layout: dict or function
    :returns: None but draws the matplotlib plt is updated and possibly saved
    :rtype: None
    """
    import networkx as nx
    if layout is None:
        layout = nx.planar_layout
    from kosh.exec_graphs import find_network_ends
    if not isinstance(layout, dict):
        layout = layout(G)

    if isinstance(G, kosh.exec_graphs.KoshExecutionGraph):
        G = G.execution_graph()
    lbls_dict = gen_labels(G)
    nx.draw(
        G,
        pos=layout,
        with_labels=True,
        labels=lbls_dict,
        alpha=.5,
        node_size=150,
        edge_color='black',
        style="dashed")
    labels = nx.get_edge_attributes(G, 'weight')
    for k in labels:
        labels[k] = "{:.3g}".format(labels[k])
    nx.draw_networkx_edge_labels(G, layout, edge_labels=labels)
    if output_format is not None:
        starters = find_network_ends(G, start=True, end=False)
        for start in starters:
            pth = nx.shortest_path(
                G, start, (output_format, None, G.seed), weight="weight")
            # build edges
            edges = []
            for i in range(len(pth) - 1):
                edges.append((pth[i], pth[i + 1]))
            nx.draw(
                G,
                pos=layout,
                with_labels=True,
                labels=lbls_dict,
                nodelist=pth,
                edgelist=edges,
                edge_color='red')
    try:
        if "DISPLAY" not in os.environ or os.environ["DISPLAY"] == "":
            import matplotlib
            matplotlib.use("agg", force=True)
        import matplotlib.pyplot as plt
        plt.show()
        plt.savefig(png_name)
        if clear:
            plt.clf()
    except ImportError:
        raise RuntimeError(
            "Could not import matplotlib, will not plot anything")


def compute_fast_sha(uri, n_samples=10):
    """Compute a fast 'almost' unique identifier for a given uri
    Assumes the uri is a path to a file, otherwise simply return hexdigest of md5 on the uri string

    If uri path is valid the 'fast' sha is used by creating an hashlib from
    * file size
    * file first 2kb
    * file last 2kb
    * 2k samples read from `n_samples` evenly spaced in the file

    Warning if size is unchanged and data is changed somewhere else than those samples the sha will be identical
    :param uri: URI to compute fast_sha on
    :type uri: str
    :param n_samples: Number of samples to extract from uri (in addition to beg and end of file)
    :type n_sampe: int
    :return sha: hexdigested sha
    :rtype: str
    """
    import numpy
    if not os.path.exists(uri):
        sha = hashlib.sha256(uri.encode())
        return sha.hexdigest()
    with open(uri, "rb") as f:
        stats = os.fstat(f.fileno())
        size = stats.st_size
        sha = hashlib.sha256("{}".format(size).encode())
        # Create list of start read
        positions = [int(max(x, 0))
                     for x in numpy.linspace(0, size - 2048, n_samples + 2)]
        prev = -1
        for pos in positions:
            # Small file will have multiple times the same bit to read
            if pos != prev:
                # Go there
                f.seek(pos)
                # read some small chunk
                st = f.read(2048)
                prev = pos
            sha.update(st)
    return sha.hexdigest()


def compute_long_sha(uri, buff_size=65536):
    """ Computes sha for a given uri
    :param uri: URI to compute fast_sha on
    :type uri: str
    :param buff_size: How much data to read at once
    :type buff_size: int
    :return sha: hexdigested sha
    :rtype: str
    """
    sha = hashlib.sha256()

    with open(uri, "rb") as f:
        while True:
            st = f.read(buff_size)
            if not st:
                break
                sha.update(st)
    return sha.hexdigest()


def update_store_record(records, store_record):
    """Updates the store record
    :param records: The sina store "records" object
    :type records: sina.datastore.DataStore.RecordOperations
    :param store_record: sina record for store info
    :type store_record: Record
    """
    if hasattr(records, "insert"):
        try:
            records.update(store_record)
        except Exception:  # in case multi-processors interfere with each others
            pass
    return


@lock_strategies.lock_function
def get_store_info_record_attribute(records, attribute):
    """Obtain an attribute from the store record info.
Sometimes when multiple process try to initiate that record it comes back with missing attributes
This will wrap into the lock strategy
    :param records: The sina store "records" object
    :type records: sina.datastore.DataStore.RecordOperations
    :param attribute: The attribute to retrievr
    :type attribute: str
    :returns: sina record for store info
    :rtype: object
    """
    rec = get_store_info_record(records)
    return rec["user_defined"][attribute]


@lock_strategies.lock_function
def get_store_info_record(records):
    """Obtain the sina record containing store info
    If necessary update store to latest standards
    :param records: The sina store "records" object
    :type records: sina.datastore.DataStore.RecordOperations
    :returns: sina record for store info
    :rtype: Record
    """
    from sina.model import Record
    # First let's see if this store contains a dedicated record
    # describing this store specs
    store_info = list(records.find_with_type("__kosh_storeinfo__"))
    if len(store_info) > 1:
        # There is a small chance that the store was created on multiple processors
        # simultaneously and that these are identical, let's try to recover
        # that was true for a small period in Kosh dev branch
        rec = store_info[0]
    elif len(store_info) == 0:
        # ok it's the old type or a new store, let's try to upgrade it for next time
        # and add the store info
        # Because of mpi ranks issues let's fix the id
        rec = Record(id="__kosh_store_info__", type="__kosh_storeinfo__", user_defined={'kosh_information': {}})
        if hasattr(records, "insert"):  # Readonly can't insert
            # It's possible many ranks will try to create this record
            # They are all identical, let's allow the error
            try:
                records.insert(rec)
            except Exception:
                pass
    else:
        rec = store_info[0]
        # This will fail if we get to version x.10
        # revisit then...
        ver = sum(
            [float(x) / 10**i for i, x in enumerate(version().split(".")) if x[0] != 'g'])
        try:
            min_ver = rec["data"]["kosh_min_version"]["value"]
        except Exception:  # Ok rec is somehow missing this info, setting to 0
            min_ver = "0.0"
        min_ver = sum(
            [float(x) / 10**i for i, x in enumerate(min_ver.split("."))])
        if ver < min_ver:
            raise RuntimeError(
                "This Kosh store requires Kosh version greater than {}, you have {}".format(min_ver, version()))
    return rec


def update_store_and_get_info_record(records, ensemble_predicate=None):
    """Obtain the sina record containing store info
    If necessary update store to latest standards
    :param records: The sina store "records" object
    :type records: sina.datastore.DataStore.RecordOperations
    :param ensemble_predicate: The predicate for the relationship to an ensemble
    :type ensemble_predicate: str
    :returns: sina record for store info
    :rtype: Record
    """
    # First let's see if this store contains a dedicated record
    # describing this store specs
    rec = get_store_info_record(records)
    need_update = False
    if "sources_type" not in rec["data"]:
        rec.add_data("sources_type", "file")
        need_update = True
    if "users_type" not in rec["data"]:
        rec.add_data("users_type", "user")
        need_update = True
    if "groups_type" not in rec["data"]:
        rec.add_data("groups_type", "group")
        need_update = True
    if "loaders_type" not in rec["data"]:
        rec.add_data("loaders_type", "koshloader")
        need_update = True
    if "ensembles_type" not in rec["data"]:
        rec.add_data("ensembles_type", "kosh_ensemble")
        need_update = True
    if "ensemble_predicate" not in rec["data"]:
        if ensemble_predicate is None:
            rec.add_data("ensemble_predicate", "is a member of ensemble")
        else:
            rec.add_data("ensemble_predicate", ensemble_predicate)
        need_update = True
    if "cached_features" not in rec["user_defined"]:
        rec["user_defined"]["cached_features"] = collections.OrderedDict()
        need_update = True
    if "kosh_min_version" not in rec["data"]:
        rec.add_data("kosh_min_version", "1.2.1")
        need_update = True
    if "reserved_types" not in rec["data"]:
        rec.add_data("reserved_types", [
            "__kosh_storeinfo__", "file", "user", "group", "kosh_ensemble", "koshloader"])
        need_update = True
    if sorted(rec["data"]["reserved_types"]["value"]) != ['__kosh_storeinfo__', 'file',
                                                          'group', 'kosh_ensemble',
                                                          'koshloader', 'user']:
        rec["data"]["reserved_types"]["value"] = ['__kosh_storeinfo__',
                                                  'file', 'group', 'kosh_ensemble', 'koshloader', 'user']
        need_update = True
    if need_update:
        update_store_record(records, rec)
    return rec


def _user_record_exists(record_handler, user_type, user, uid=None):
    """Check whether a Kosh user record already exists.

    :param record_handler: The sina records object.
    :type record_handler: sina.records
    :param user_type: The Sina record type used for users.
    :type user_type: str
    :param user: Username to look up.
    :type user: str
    :param uid: Optional deterministic user id for direct lookup.
    :type uid: str or None
    :returns: ``True`` when the user record is already present.
    :rtype: bool
    """
    existing = list(record_handler.find(types=[user_type], data={"username": user}))
    if existing:
        return True
    if uid is None:
        return False
    try:
        record_handler.get(uid)
        return True
    except Exception:
        return False


def create_kosh_users(record_handler, users=None):
    """Add Kosh users to the Kosh store.

    This function is safe to call concurrently. If another process inserts the
    same deterministic user record between the existence check and the insert,
    the duplicate-key error is treated as success once the record is confirmed
    to exist.

    :param record_handler: The sina records object.
    :type record_handler: sina.records
    :param users: Usernames to ensure exist in the store.
    :type users: list or None
    """
    from sina.model import Record
    if users is None:
        users = [os.environ.get("USER", "default"), "anonymous"]
    store_info = list(record_handler.find_with_type(
        ["__kosh_storeinfo__", ]))[0]

    user_type = store_info["data"]["users_type"]["value"]
    # Create users
    for user in users:
        uid = hashlib.md5(user.encode()).hexdigest()
        if not _user_record_exists(record_handler, user_type, user, uid=uid):
            user_record = Record(id=uid, type=user_type, user_defined={'kosh_information': {}})
            user_record.add_data("username", user)
            try:
                record_handler.insert(user_record)
            except Exception:
                if not _user_record_exists(record_handler, user_type, user, uid=uid):
                    raise


def create_new_db(name, db='sql',
                  keyspace=None, **kargs):
    """create_new_db creates a new Kosh database, adds a single user

    :param name: name of database
    :type name: str
    :param db: type of database, defaults to 'sql', can be 'cass'
    :type db: str, optional
    :param keyspace: for cassandra keyspace to use, defaults to None means [user]_k
    :type keyspace: str, optional
    :param kargs: Any additional key/value pairs you need to pass to store creation
    :type kargs: dict
    :return store: An handle to the Kosh store created
    :rtype: KoshStoreClass
    """
    if os.path.exists(name):
        os.remove(name)
    from kosh import connect
    kargs["keyspace"] = keyspace
    kargs["db"] = db
    # Let's remove the now unused arguments
    for key in ["token", "cluster"]:
        if key in kargs:
            warnings.warn(
                "Keyword '{}' is no longer valid, will be ignored".format(key))
            kargs.pop(key)
    store = connect(name, **kargs)
    store.delete_all_contents(force="SKIP PROMPT")
    return store


def version(comparable=False):
    """Returns version string
    :param comparable: returns version as a tuple of ints so it can be compared
    :type comparable: bool
    :returns: version string or tuple
    :rtype: str or tuple
    """
    try:
        __version__ = importlib.metadata.version("kosh")
    except importlib.metadata.PackageNotFoundError:
        __version__ = "0.0"
    if comparable:
        tuple_version = ()
        for number in __version__.split("."):
            try:
                tuple_version += (int(number),)
            except ValueError:  # Probably some letter or symbol in here
                pass
        __version__ = tuple_version
    return __version__


def walk_dictionary_keys(dictionary, separator="/"):
    """Walks through a dictionary and return all levels of keys
    sub dictionary keys are append to parent key with the 'separator'
    :param dictionary: The dictionary to walk
    :type dictionary: dict
    :param separator: The string to use between a parent key and its children
    :type separator: str
    :returns: generator of keys and possibly their sub keys
    :rtype: generator
    """
    out = []
    for key in sorted(dictionary.keys(), key=lambda x: str(x)):
        out.append(str(key))
        if isinstance(dictionary[key], dict):
            yld = walk_dictionary_keys(dictionary[key], separator)
            for y in yld:
                st = "{}{}{}".format(key, separator, y)
                out.append(st)
    return out


def get_graph(input_type, loader, transformers):
    """Given a loader and its transformer return path to desired format
    e.g which output format should each transformer pick to be chained to the following one
    in order to obtain the desired outcome for format
    :param input_type: input type of first node
    :type input_type: str
    :param loader: original loader
    :type loader: KoshLoader
    :param transformers: set of transformers to be added after loader exits
    :type transformers: list of KoshTransformer
    :returns: execution graph
    :rtype: networkx.OrderDiGraph
    """
    import networkx as nx
    from kosh.exec_graphs import populate
    if input_type not in loader.types:
        raise RuntimeError(
            "loader cannot load mime_type {}".format(input_type))
    try:
        G = nx.OrderedDiGraph()
    except AttributeError:  # networkx 3.0 removed OrderedDiGraph
        G = nx.DiGraph()
    G.seed = random.random()
    start_node = (input_type, loader, G.seed)  # so each graph is unique
    G.add_node(start_node)
    if len(transformers) == 0:
        # No transformer
        for out_format in loader.types[input_type]:
            node = (out_format, None, G.seed)
            G.add_edge(start_node, node)
    else:
        populate(
            G,
            start_node,
            loader.types[input_type],
            transformers)
    return G


def cleanup_sina_record_from_kosh_sync(record):
    """Kosh adds data in the 'user_defined' section of records to keep track of syncing
    This removes these attributes
    :param record: The Sina record to cleanup
    :type record: sina.model.Record
    :return: json loaded representation of the record
    :rtype: dict"""
    # cleanup the record
    for key in list(record["user_defined"]['kosh_information'].keys()):
        if key.endswith("last_modified"):
            record["user_defined"]['kosh_information'].pop(key)
    return orjson.loads(record.to_json())


def record_to_dataset(record):
    """Converts a Sina record to a KoshDatset
    :param record: The Sina Record to convert
    :type record: sina.model.Record
    :return: kosh version of the record
    :rtype: kosh.KoshDatset"""
    temp_store = kosh.connect(None)
    temp_store.get_sina_records().insert(record)
    return next(temp_store.find())


def record_to_dataframe(rec_uri):
    """Converts a Sina record to a Pandas DataFrame
    :param rec_uri: The Sina Record URI to convert
    :type rec_uri: str
    :return: Pandas DataFrame version of the record
    :rtype: Pandas.DataFrame"""
    temp_store = kosh.connect(None)
    temp_store.import_dataset(rec_uri, match_attributes=['id'])
    return temp_store.to_dataframe()


def datasets_in_place_of_records(func):
    """This decorator will convert all Record input or output to KoshDataset
    This allows a user to use sina functions that expect Record with Kosh datasets instead"""

    from sina.model import Record

    def wrapper(*args, **kwargs):
        new_args = [x.get_record() if isinstance(x, kosh.KoshDataset) else x for x in args]
        new_kwargs = {}
        for k, v in kwargs.items():
            new_kwargs[k] = v.get_record() if isinstance(v, kosh.KoshDataset) else v
        out = func(*new_args, **new_kwargs)
        if hasattr(out, "__iter__"):
            return [record_to_dataset(v) if isinstance(v, Record) else v for v in out]
        else:
            return record_to_dataset(out) if isinstance(out, Record) else out
    return wrapper


def update_json_file_with_records_and_relationships(file, output_dict):
    if file is not None:
        if os.path.exists(file):
            with open(file) as f:
                file_dict = orjson.loads(f.read())
            records_already_in_file = file_dict["records"]
            # You can't "set" records
            # This erased records
            file_dict.update(output_dict)
            # ids that are now in file_dict
            new_dataset_rec_ids = [x["id"] for x in output_dict["records"]]
            # Make sure we put the records that were in the file back in
            for record in records_already_in_file:
                if record["id"] not in new_dataset_rec_ids:
                    file_dict["records"].append(record)
        else:
            file_dict = output_dict

        with open(file, "w") as f:
            rels = []
            for relationship in file_dict['relationships']:
                rels.append(relationship.__dict__)
            file_dict['relationships'] = rels
            try:
                f.write(orjson.dumps(file_dict).decode())
            except AttributeError:
                f.write(orjson.dumps(file_dict))
    return output_dict


def __check_valid_connection_type__(connection_type, connection_types_allowed):
    """Checks if the `connection_type` is in `connection_types_allowed` for different stores

    :param connection_types_allowed: The connection types allowed for this store type
    :type connection_types_allowed: lst
    """
    if connection_type not in connection_types_allowed:
        raise RuntimeError(f"This functionality is allowed for {connection_types_allowed} only stores.")


def _update_record(records, kosh_store, sina_store=None, delete=False):
    """Central location to insert or update or delete a record in the sina store
    :param record: The record(s) to update
    :type record: sina.model.Record
    :param kosh_store: kosh store to lock so that sina store can update
    :type kosh_store: Kosh store
    :param sina_store: sina store to update
    :type sina_store: sina.datastore.DataStore"""
    kosh_store.lock()

    if sina_store is None:
        sina_store = kosh_store.get_sina_store()

    if not isinstance(records, list):
        records = [records]

    for record in records:

        try:
            if isinstance(record, str):  # just record id
                record = kosh_store.get_record(record)

            id_ = record.id
        except ValueError:  # unsynced delete won't be in database
            id_ = record

        if delete:
            sina_store.records.delete(id_)
        else:
            rels = sina_store.relationships.find(id_, None, None)
            rels += sina_store.relationships.find(None, None, id_)
            if not kosh_store.__sync__:
                sina_store.records.delete(id_)
                sina_store.records.insert(record)
                sina_store.relationships.insert(rels)
            record.set_data("last_modified_date", str(datetime.fromtimestamp(time.time())))

            try:
                sina_store.records.update(record)

            # record does not exist
            except:  # noqae722
                sina_store.records.insert(record)

    kosh_store.unlock()
