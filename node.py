# Copyright (C) 2024 Cyprien Quéméneur
# FedPylot is released under the GPL-3.0 license, please refer to the LICENSE file in the root directory of the program.
# For the full copyright notices, please refer to the NOTICE file in the root directory of the program.

import copy
import os
import pickle
import secrets
import sys

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from mpi4py import MPI
import torch
import yaml

from models.yolo import Model
from utils.torch_utils import intersect_dicts, is_parallel, select_device


class Node:
    """General node logic common to the server and clients. Evaluation is performed here to allow personalized FL."""

    def __init__(self, rank: int) -> None:
        """Initialize the node with its rank, device, public and private keys, and symmetric key password."""
        self.rank = rank
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._ckpt = None
        self._ckpt_reparam = None
        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self._public_key = self._private_key.public_key()
        self._symmetric_key = None
        self._password = b'my great password'

    @property
    def public_key(self) -> bytes:
        """Return the serialized public key of the node."""
        serialized_public_key = self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return serialized_public_key

    def get_device_info(self) -> None:
        """Print the device information."""
        if self.device == 'cuda':
            device_id = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_id)
            print(f'Node of rank {self.rank}: '
                  f'Found {torch.cuda.device_count()} GPU(s) available. '
                  f'Using GPU {device_id} ({properties.name}) '
                  f'of compute capability {properties.major}.{properties.minor} with '
                  f'{properties.total_memory / 1e9:.1f}Gb total memory.')
        else:
            print(f'Node of rank {self.rank}: Using CPU for training.')

    def _symmetric_encryption(self, data_to_encrypt: dict) -> tuple[bytes, bytes, bytes]:
        """Encrypt the checkpoint, weights or update using the node's symmetric key with AES-GCM."""
        serialized_data = pickle.dumps(data_to_encrypt)
        nonce = secrets.token_bytes(12)  # 96-bit for the nonce for AES-GCM
        cipher = Cipher(algorithms.AES(self._symmetric_key), modes.GCM(nonce))
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(serialized_data) + encryptor.finalize()
        tag = encryptor.tag
        return encrypted_data, tag, nonce

    def _symmetric_decryption(self, data_to_decrypt: bytes, tag: bytes, nonce: bytes) -> dict:
        """Decrypt the checkpoint, weights or update using the node's symmetric key with AES-GCM."""
        cipher = Cipher(algorithms.AES(self._symmetric_key), modes.GCM(nonce, tag))
        decryptor = cipher.decryptor()
        serialized_data = decryptor.update(data_to_decrypt) + decryptor.finalize()
        return pickle.loads(serialized_data)

    def _asymmetric_decryption(self, data_to_decrypt: bytes) -> bytes:
        """Decrypt the symmetric key using the node's private key with RSA-OAEP."""
        data_decrypted = self._private_key.decrypt(
            data_to_decrypt,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return data_decrypted

    def reparameterize(self, architecture: str = 'yolov7') -> None:
        """Reduce trainable Bag of Freebies modules into deploy model for faster inference."""
        ckpt = copy.deepcopy(self._ckpt)
        backup_hyp = ckpt['model'].hyp
        backup_gr = ckpt['model'].gr
        nc = ckpt['model'].nc
        deploy_path = f'yolov7/cfg/deploy/{architecture}.yaml'
        id_mp = {
            'yolov7-tiny': 77,
            'yolov7': 105,
            'yolov7x': 121,
            'yolov7-w6': 118,
            'yolov7-e6': 140,
            'yolov7-d6': 162,
            'yolov7-e6e': 261
        }
        model = Model(deploy_path, ch=3, nc=nc).to(self.device)
        with open(deploy_path) as f:
            yml = yaml.load(f, Loader=yaml.SafeLoader)
        anchors = len(yml['anchors'][0]) // 2
        sd = ckpt['model'].float().state_dict()
        exclude = []
        intersect_state_dict = {k: v for k, v in sd.items() if
                                k in model.state_dict() and not any(x in k for x in exclude)
                                and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(intersect_state_dict, strict=False)
        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc
        if architecture in ['yolov7-tiny', 'yolov7', 'yolov7x']:
            # Re-parameterization of P5 models
            idx = id_mp[architecture]
            for i in range((model.nc + 5) * anchors):
                model.state_dict()[f'model.{idx}.m.0.weight'].data[i, :, :, :] *= sd[f'model.{idx}.im.0.implicit'].data[:, i, ::].squeeze()
                model.state_dict()[f'model.{idx}.m.1.weight'].data[i, :, :, :] *= sd[f'model.{idx}.im.1.implicit'].data[:, i, ::].squeeze()
                model.state_dict()[f'model.{idx}.m.2.weight'].data[i, :, :, :] *= sd[f'model.{idx}.im.2.implicit'].data[:, i, ::].squeeze()
            model.state_dict()[f'model.{idx}.m.0.bias'].data += sd[f'model.{idx}.m.0.weight'].mul(sd[f'model.{idx}.ia.0.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{idx}.m.1.bias'].data += sd[f'model.{idx}.m.1.weight'].mul(sd[f'model.{idx}.ia.1.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{idx}.m.2.bias'].data += sd[f'model.{idx}.m.2.weight'].mul(sd[f'model.{idx}.ia.2.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{idx}.m.0.bias'].data *= sd[f'model.{idx}.im.0.implicit'].data.squeeze()
            model.state_dict()[f'model.{idx}.m.1.bias'].data *= sd[f'model.{idx}.im.1.implicit'].data.squeeze()
            model.state_dict()[f'model.{idx}.m.2.bias'].data *= sd[f'model.{idx}.im.2.implicit'].data.squeeze()
        else:
            # Re-parameterization of P6 models
            idx = id_mp[architecture]
            idx2 = idx + 4
            model.state_dict()[f'model.{idx}.m.0.weight'].data -= model.state_dict()[f'model.{idx}.m.0.weight'].data
            model.state_dict()[f'model.{idx}.m.1.weight'].data -= model.state_dict()[f'model.{idx}.m.1.weight'].data
            model.state_dict()[f'model.{idx}.m.2.weight'].data -= model.state_dict()[f'model.{idx}.m.2.weight'].data
            model.state_dict()[f'model.{idx}.m.3.weight'].data -= model.state_dict()[f'model.{idx}.m.3.weight'].data
            model.state_dict()[f'model.{idx}.m.0.weight'].data += sd[f'model.{idx2}.m.0.weight'].data
            model.state_dict()[f'model.{idx}.m.1.weight'].data += sd[f'model.{idx2}.m.1.weight'].data
            model.state_dict()[f'model.{idx}.m.2.weight'].data += sd[f'model.{idx2}.m.2.weight'].data
            model.state_dict()[f'model.{idx}.m.3.weight'].data += sd[f'model.{idx2}.m.3.weight'].data
            model.state_dict()[f'model.{idx}.m.0.bias'].data -= model.state_dict()[f'model.{idx}.m.0.bias'].data
            model.state_dict()[f'model.{idx}.m.1.bias'].data -= model.state_dict()[f'model.{idx}.m.1.bias'].data
            model.state_dict()[f'model.{idx}.m.2.bias'].data -= model.state_dict()[f'model.{idx}.m.2.bias'].data
            model.state_dict()[f'model.{idx}.m.3.bias'].data -= model.state_dict()[f'model.{idx}.m.3.bias'].data
            model.state_dict()[f'model.{idx}.m.0.bias'].data += sd[f'model.{idx2}.m.0.bias'].data
            model.state_dict()[f'model.{idx}.m.1.bias'].data += sd[f'model.{idx2}.m.1.bias'].data
            model.state_dict()[f'model.{idx}.m.2.bias'].data += sd[f'model.{idx2}.m.2.bias'].data
            model.state_dict()[f'model.{idx}.m.3.bias'].data += sd[f'model.{idx2}.m.3.bias'].data
            for i in range((model.nc + 5) * anchors):
                model.state_dict()[f'model.{idx}.m.0.weight'].data[i, :, :, :] *= sd[f'model.{idx2}.im.0.implicit'].data[:, i, : :].squeeze()
                model.state_dict()[f'model.{idx}.m.1.weight'].data[i, :, :, :] *= sd[f'model.{idx2}.im.1.implicit'].data[:, i, : :].squeeze()
                model.state_dict()[f'model.{idx}.m.2.weight'].data[i, :, :, :] *= sd[f'model.{idx2}.im.2.implicit'].data[:, i, : :].squeeze()
                model.state_dict()[f'model.{idx}.m.3.weight'].data[i, :, :, :] *= sd[f'model.{idx2}.im.3.implicit'].data[:, i, : :].squeeze()
            model.state_dict()[f'model.{idx}.m.0.bias'].data += sd[f'model.{idx2}.m.0.weight'].mul(sd[f'model.{idx2}.ia.0.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{idx}.m.1.bias'].data += sd[f'model.{idx2}.m.1.weight'].mul(sd[f'model.{idx2}.ia.1.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{idx}.m.2.bias'].data += sd[f'model.{idx2}.m.2.weight'].mul(sd[f'model.{idx2}.ia.2.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{idx}.m.3.bias'].data += sd[f'model.{idx2}.m.3.weight'].mul(sd[f'model.{idx2}.ia.3.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{idx}.m.0.bias'].data *= sd[f'model.{idx2}.im.0.implicit'].data.squeeze()
            model.state_dict()[f'model.{idx}.m.1.bias'].data *= sd[f'model.{idx2}.im.1.implicit'].data.squeeze()
            model.state_dict()[f'model.{idx}.m.2.bias'].data *= sd[f'model.{idx2}.im.2.implicit'].data.squeeze()
            model.state_dict()[f'model.{idx}.m.3.bias'].data *= sd[f'model.{idx2}.im.3.implicit'].data.squeeze()
        # Saving re-parameterized model
        model.hyp = backup_hyp
        model.gr = backup_gr
        self._ckpt_reparam = {'model': copy.deepcopy(model.module if is_parallel(model) else model).half(),
                              'optimizer': None,
                              'training_results': None,
                              'epoch': -1}

    def post_init_update(self, data: str, cfg: str, hyp: str, imgsz: int) -> None:
        """Post-initialization update of the model to match the training loop model."""
        ckpt = copy.deepcopy(self._ckpt)  # load checkpoint
        with open(data) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)
        with open(hyp) as f:
            hyp_dict = yaml.load(f, Loader=yaml.SafeLoader)
        nc = int(data_dict['nc'])  # number of classes
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp_dict.get('anchors'))
        model = model.to(self.device)
        exclude = ['anchor'] if (cfg or hyp_dict.get('anchors')) else []
        state_dict = ckpt['model'].float().state_dict()
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)
        # Model parameters
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        hyp_dict['box'] *= 3. / nl  # scale to layers
        hyp_dict['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp_dict['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        hyp_dict['label_smoothing'] = 0.0
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp_dict  # attach hyperparameters to model
        model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        model.names = data_dict['names']  # attach class names to model
        self._ckpt['model'] = model

    def test(self, kround: int, saving_path: str, data: str, bsz: int, imgsz: int, conf: float, iou: float) -> None:
        """Evaluate the model on the validation set held by the node."""
        weights = f'{saving_path}/weights/eval-kround{kround}.pt'
        torch.save(self._ckpt_reparam, weights)
        os.system(
            f'python ./yolov7/test.py'
            f' --kround {kround}'
            f' --saving-path {saving_path}'
            f' --weights {weights}'
            f' --data {data}'
            f' --batch {bsz}'
            f' --img {imgsz}'
            f' --conf-thres {conf}'
            f' --iou-thres {iou}'
            f' --task val'
            f' --project {saving_path}/run/'
            f' --name eval-kround{kround}'
            f' --no-trace'
        )


class Server(Node):
    """Specific server logic (model initialization, server-side optimization, weights and key sharing)."""

    def __init__(self, server_opt: str = 'fedavg', serverlr: float = 1., tau: float = None, beta: float = None) -> None:
        """Initialize the server with rank 0 and optimizer (fedavg, fedavgm, fedadagrad or fedadam)."""
        super().__init__(rank=0)
        self.server_opt = server_opt
        self.server_lr = serverlr
        self.__clients_public_keys = None
        # FedAvgM additional parameters
        if self.server_opt == 'fedavgm':
            self.beta = beta
            self.v_t = None
        # FedAdagrad additional parameters
        if self.server_opt == 'fedadagrad':
            self.tau = tau
            self.v_t = None
        # FedAdam additional parameters
        if self.server_opt == 'fedadam':
            self.beta1 = 0.9
            self.beta2 = 0.99
            self.tau = tau
            self.m_t = None
            self.v_t = None

    @property
    def clients_public_keys(self):
        """Return the serialized public keys of the clients."""
        return self.__clients_public_keys

    @clients_public_keys.setter
    def clients_public_keys(self, serialized_keys: dict) -> None:
        """Un-serialize the public keys of the clients and store them in a dictionary."""
        self.__clients_public_keys = {r: load_pem_public_key(serialized_keys[r]) for r in serialized_keys.keys()}

    def generate_symmetric_key(self) -> None:
        """Generate a symmetric key using the node's password with Scrypt."""
        key_length = 32  # AES-256 key length (256 bits)
        salt = secrets.token_bytes(key_length)
        kdf = Scrypt(salt=salt, length=key_length, n=2 ** 20, r=8, p=1)
        symmetric_key = kdf.derive(self._password)
        self._symmetric_key = symmetric_key

    def get_symmetric_key(self) -> list[bytes]:
        """Return a list composed of the symmetric key encrypted with each client's public key."""
        sks_encrypted = []
        for client_rank in self.__clients_public_keys.keys():
            public_key = self.__clients_public_keys[client_rank]
            sk_encrypted = public_key.encrypt(
                self._symmetric_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            sks_encrypted.append(sk_encrypted)
        return sks_encrypted

    def initialize_model(self, weights: str) -> None:
        """Initialize the checkpoint from a pretrained weights file."""
        self._ckpt = torch.load(weights, map_location=self.device)

    def get_weights(self, metadata: bool) -> list[tuple[bytes, bytes, bytes]]:
        """Return the weights encrypted with AES, the tag, and the nonce for each client."""
        weights = copy.deepcopy(self._ckpt) if metadata else copy.deepcopy(self._ckpt['model']).half().state_dict()
        encrypted_weights, tag, nonce = self._symmetric_encryption(weights)
        encrypted_data = []
        for client_rank in self.__clients_public_keys.keys():
            encrypted_data.append((encrypted_weights, tag, nonce))
            print(f'Communication cost server-side for client{client_rank}: '
                  f' - encrypted weights size (metadata={metadata}): {sys.getsizeof(encrypted_weights)}'
                  f' - total cost of one server to node communication with MPI:'
                  f' {sys.getsizeof(MPI.pickle.dumps((encrypted_weights, tag, nonce)))}')
        return encrypted_data

    def __decrypt_updates(self, sds_encrypted: list[tuple[bytes, bytes, bytes, int]]) -> tuple[list[dict], list[int]]:
        """Decrypt the encrypted updates with the symmetric key."""
        state_dicts = []
        nsamples_list = []
        for k in range(len(sds_encrypted)):
            sd_encrypted, tag, nonce, nsamples = sds_encrypted[k]
            state_dict = self._symmetric_decryption(sd_encrypted, tag, nonce)
            state_dicts.append(state_dict)
            nsamples_list.append(nsamples)
        return state_dicts, nsamples_list

    def __compute_pseudo_gradient(self, updates: list[dict], nsamples_list: list[int]) -> dict:
        """Compute the pseudo-gradient using the weighted average of the updates received from the clients."""
        n = 0
        for ni in nsamples_list:
            n += ni
        delta_t = copy.deepcopy(self._ckpt['model'].state_dict())
        for key in delta_t.keys():
            delta_it_weighted = [delta_it[key] * (ni / n) for delta_it, ni in zip(updates, nsamples_list)]
            delta_t[key] = torch.sum(torch.stack(delta_it_weighted), dim=0)
        return delta_t

    def __fedavg(self, delta_t: dict) -> dict:
        """Compute the new weights using the FedAvg algorithm (server opt is SGD, default lr is 1)."""
        w_t = copy.deepcopy(self._ckpt['model'].state_dict())
        for key in w_t.keys():
            w_t[key] = w_t[key] - self.server_lr * delta_t[key]  # SGD with pseudo-gradient
        return w_t

    def __fedavgm(self, delta_t: dict) -> dict:
        """Compute the new weights using the FedAvgM algorithm (server opt is SGD with momentum, default lr is 1)."""
        w_t = copy.deepcopy(self._ckpt['model'].state_dict())
        self.v_t = {key: self.beta * self.v_t[key] + delta_t[key] for key in delta_t.keys()}
        w_t = {key: w_t[key] - self.server_lr * self.v_t[key] for key in delta_t.keys()}
        return w_t

    def __fedadagrad(self, delta_t):
        """Compute the new weights using the FedAdagrad algorithm (server opt is Adagrad)."""
        w_t = copy.deepcopy(self._ckpt['model'].state_dict())
        self.v_t = {key: self.v_t[key] + delta_t[key] ** 2 for key in delta_t.keys()}
        w_t = {
            key: w_t[key] - self.server_lr * delta_t[key] / (torch.sqrt(self.v_t[key]) + self.tau)
            for key in delta_t.keys()
        }
        return w_t

    def __fedadam(self, delta_t):
        """Compute the new weights using the FedAdam algorithm (server opt is Adam with default decay parameters)."""
        w_t = copy.deepcopy(self._ckpt['model'].state_dict())
        self.m_t = {
            key: self.beta1 * self.m_t[key] + (1. - self.beta1) * delta_t[key]
            for key in delta_t.keys()
        }
        self.v_t = {
            key: self.beta2 * self.v_t[key] + (1. - self.beta2) * delta_t[key] ** 2
            for key in delta_t.keys()
        }
        w_t = {
            key: w_t[key] - self.server_lr * self.m_t[key] / (torch.sqrt(self.v_t[key]) + self.tau)
            for key in delta_t.keys()
        }
        return w_t

    def aggregate(self, state_dicts_encrypted: list[tuple[bytes, bytes, bytes, int]]) -> None:
        """Compute the weights for the next communication round using the clients' local updates."""
        updates, nsamples_list = self.__decrypt_updates(state_dicts_encrypted)
        delta_t = self.__compute_pseudo_gradient(updates, nsamples_list)
        if self.server_opt == 'fedavg':
            new_sd = self.__fedavg(delta_t)
        elif self.server_opt == 'fedavgm':
            if self.v_t is None:
                self.v_t = {key: torch.zeros_like(delta_t[key]) for key in delta_t.keys()}
            new_sd = self.__fedavgm(delta_t)
        elif self.server_opt == 'fedadagrad':
            if self.v_t is None:
                self.v_t = {key: torch.zeros_like(delta_t[key]) for key in delta_t.keys()}
            new_sd = self.__fedadagrad(delta_t)
        elif self.server_opt == 'fedadam':
            if self.m_t is None:
                self.m_t = {key: torch.zeros_like(delta_t[key]) for key in delta_t.keys()}
            if self.v_t is None:
                self.v_t = {key: torch.zeros_like(delta_t[key]) for key in delta_t.keys()}
            new_sd = self.__fedadam(delta_t)
        else:
            raise ValueError('Server optimizer not recognized, must be fedavg, fedavgm, fedadagrad, or fedadam')
        model = self._ckpt['model']
        model.load_state_dict(new_sd)
        self._ckpt['model'] = model


class Client(Node):
    """Specific client logic (local training, update and key sharing)."""

    def __init__(self, rank: int) -> None:
        """Initialize the client with its rank."""
        if rank <= 0:
            raise ValueError('Rank 0 is reserved for the MPI orchestrator / federated server.')
        super().__init__(rank)
        self.__server_public_key = None
        self.__update = None
        self.nsamples = None

    @property
    def server_public_key(self):
        """Return the serialized public key of the server."""
        return self.__server_public_key

    @server_public_key.setter
    def server_public_key(self, serialized_key: bytes) -> None:
        """Un-serialize the public key of the server and store it."""
        self.__server_public_key = load_pem_public_key(serialized_key)

    @property
    def symmetric_key(self) -> None:
        """Block access to the unencrypted symmetric key."""
        print('The symmetric key is private.')
        return None

    @symmetric_key.setter
    def symmetric_key(self, sk_encrypted: bytes) -> None:
        """Decrypt the symmetric key using the node's private key and store it."""
        self._symmetric_key = self._asymmetric_decryption(sk_encrypted)

    def get_update(self) -> tuple[bytes, bytes, bytes, int]:
        """Return the encrypted update (AES), tag, nonce, and the number of local training examples."""
        encrypted_update, tag, nonce = self._symmetric_encryption(self.__update)
        print(f'Communication cost for client{self.rank}:'
              f' - encrypted update size: {sys.getsizeof(encrypted_update)}'
              f' - total cost of one node to server communication with MPI:'
              f' {sys.getsizeof(MPI.pickle.dumps((encrypted_update, tag, nonce, self.nsamples)))}')
        return encrypted_update, tag, nonce, self.nsamples

    def set_weights(self, encrypted_data: tuple[bytes, bytes, bytes], metadata: bool) -> None:
        """Decrypt the weights or checkpoint with the symmetric key and save it."""
        new_weights_encrypted, tag, nonce = encrypted_data
        new_weights = self._symmetric_decryption(new_weights_encrypted, tag, nonce)
        if metadata:
            self._ckpt = new_weights
        else:
            model = self._ckpt['model']
            model.load_state_dict(new_weights)
            self._ckpt['model'] = model

    def train(self, nrounds: int, kround: int, epochs: int, architecture: str, data: str, bsz_train: int, imgsz: int,
              cfg: str, hyp: str, workers: int, saving_path: str) -> None:
        """Train the model on the local training set and store the new checkpoint and update."""
        end_weights = f'{saving_path}/run/train-client{self.rank}/weights/last.pt'
        if architecture in ['yolov7-tiny', 'yolov7', 'yolov7x']:
            script_path = './yolov7/train.py'
        elif architecture in ['yolov7-w6', 'yolov7-e6', 'yolov7-d6', 'yolov7-e6e']:
            script_path = './yolov7/train_aux.py'
        else:
            raise ValueError(f'Model architecture {architecture} not recognized.')
        if kround == 0:
            # Initialize the training loop and perform the first round of training
            begin_weights = f'{saving_path}/weights/train-kround{kround}-client{self.rank}.pt'
            torch.save(self._ckpt, begin_weights)
            os.system(
                f'python {script_path}'
                f' --client-rank {self.rank}'
                f' --round-length {epochs}'
                f' --batch-size {bsz_train}'
                f' --epochs {nrounds * epochs}'
                f' --data {data}'
                f' --img {imgsz} {imgsz}'
                f' --cfg {cfg}'
                f' --weights {begin_weights}'
                f' --hyp {hyp}'
                f' --workers {workers}'
                f' --project {saving_path}/run/'
                f' --name train-client{self.rank}'
                f' --notest'
            )
        else:
            # Resume training with the new set of weights
            begin_weights = f'{saving_path}/run/train-client{self.rank}/weights/last.pt'
            torch.save(self._ckpt, begin_weights)
            os.system(f'python {script_path} --resume {begin_weights}')
        new_ckpt = torch.load(end_weights, map_location=self.device)
        # Compute the local update: delta_it = w_t - w_it
        w_it = new_ckpt['model'].state_dict()
        if kround == 0:
            self.post_init_update(data, cfg, hyp, imgsz)
        w_t = self._ckpt['model'].half().state_dict()
        delta_it = copy.deepcopy(w_t)
        for key in delta_it.keys():
            delta_it[key] = w_t[key] - w_it[key]
        self.__update = delta_it
        # Maintain state across communication rounds (required for FedOpt)
        self._ckpt = new_ckpt
