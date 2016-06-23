from ctypes import POINTER, create_string_buffer, pointer
import numpy as np
from mjtypes import *
from mjlib import mjlib
from util import *
import mjconstants as C


class MjError(Exception):
    pass


def register_license(file_path):
    result = mjlib.mj_activate(file_path)
    if result == 1:
        pass
    elif result == 0:
        raise MjError('could not register license')
    else:
        raise MjError("I don't know wth happene")


class MjModel(MjModelWrapper):

    def __init__(self, xml_path):
        buf = create_string_buffer(1000)
        model_ptr = mjlib.mj_loadXML(xml_path, None, buf, 1000)
        if len(buf.value) > 0:
            super(MjModel, self).__init__(None)
            raise MjError(buf.value)
        super(MjModel, self).__init__(model_ptr)
        data_ptr = mjlib.mj_makeData(model_ptr)
        data = MjData(data_ptr, self)
        self.data = data
        self._body_comvels = None
        self.forward()

    def forward(self):
        mjlib.mj_forward(self.ptr, self.data.ptr)
        self._body_comvels = None

    @property
    def body_comvels(self):
        if self._body_comvels is None:
            self._body_comvels = self._compute_subtree()
        return self._body_comvels

    def _compute_subtree(self):
        body_vels = np.zeros((self.nbody, 6))
        # bodywise quantities
        mass = self.body_mass.flatten()
        for i in xrange(self.nbody):
            # body velocity
            mjlib.mj_objectVelocity(
                self.ptr, self.data.ptr, C.mjOBJ_BODY, i,
                body_vels[i].ctypes.data_as(POINTER(c_double)), 0
            )
            # body linear momentum
        lin_moms = body_vels[:, 3:] * mass.reshape((-1, 1))

        # init subtree mass
        body_parentid = self.body_parentid
        # subtree com and com_vel
        for i in xrange(self.nbody - 1, -1, -1):
            if i > 0:
                parent = body_parentid[i]
                # add scaled velocities
                lin_moms[parent] += lin_moms[i]
                # accumulate mass
                mass[parent] += mass[i]
        return lin_moms / mass.reshape((-1, 1))

    def step(self):
        mjlib.mj_step(self.ptr, self.data.ptr)

    def __del__(self):
        if self._wrapped is not None:
            mjlib.mj_deleteModel(self._wrapped)
            self._wrapped = None

    @property
    def body_names(self):
        start_addr = ctypes.addressof(self.names.contents)
        return [ctypes.string_at(start_addr + inc)
                for inc in self.name_bodyadr.flatten()]

    @property
    def joint_names(self):
        start_addr = ctypes.addressof(self.names.contents)
        return [ctypes.string_at(start_addr + inc)
                for inc in self.name_jntadr.flatten()]

    @property
    def geom_names(self):
        start_addr = ctypes.addressof(self.names.contents)
        return [ctypes.string_at(start_addr + inc)
                for inc in self.name_geomadr.flatten()]

    @property
    def site_names(self):
        start_addr = ctypes.addressof(self.names.contents)
        return [ctypes.string_at(start_addr + inc)
                for inc in self.name_siteadr.flatten()]

    @property
    def mesh_names(self):
        start_addr = ctypes.addressof(self.names.contents)
        return [ctypes.string_at(start_addr + inc)
                for inc in self.name_meshadr.flatten()]

    @property
    def numeric_names(self):
        start_addr = ctypes.addressof(self.names.contents)
        return [ctypes.string_at(start_addr + inc)
                for inc in self.name_numericadr.flatten()]


class MjData(MjDataWrapper):

    def __init__(self, wrapped, size_src=None):
        super(MjData, self).__init__(wrapped, size_src)

    def __del__(self):
        if self._wrapped is not None:
            mjlib.mj_deleteData(self._wrapped)

    @property
    def contact(self):
        contacts = self._wrapped.contents.contact[:self.ncon]
        return [MjContactWrapper(pointer(c)) for c in contacts]
