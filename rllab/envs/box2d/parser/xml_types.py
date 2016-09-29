# pylint: disable=no-init, too-few-public-methods, old-style-class

from types import LambdaType


def _extract_type(typ):
    if isinstance(typ, LambdaType):
        return typ()
    else:
        return typ


class AttrDecl(object):
    pass


class XmlChildren(AttrDecl):

    def __init__(self, tag, typ):
        self._tag = tag
        self._typ = typ

    def from_xml(self, xml):
        xml_elems = [child for child in xml if child.tag == self._tag]
        typ = _extract_type(self._typ)
        return [typ.from_xml(elem) for elem in xml_elems]


class XmlChild(AttrDecl):

    def __init__(self, tag, typ, required=False):
        self._tag = tag
        self._typ = typ
        self._required = required

    def from_xml(self, xml):
        xml_elems = [child for child in xml if child.tag == self._tag]
        if len(xml_elems) > 1:
            raise ValueError('Multiple candidate found for tag %s' % self._tag)
        if len(xml_elems) == 0:
            if self._required:
                raise ValueError('Missing xml element with tag %s' % self._tag)
            else:
                return None
        elem = xml_elems[0]
        return _extract_type(self._typ).from_xml(elem)


class XmlAttr(AttrDecl):

    def __init__(self, name, typ, required=False, *args, **kwargs):
        self._name = name
        self._typ = typ
        self._required = required

    @property
    def name(self):
        return self._name

    def from_xml(self, xml):
        if self._name in xml.attrib:
            return _extract_type(self._typ).from_str(xml.attrib[self._name])
        elif self._required:
            raise ValueError("Missing required attribute %s" % self._name)
        else:
            return None


class XmlElem(object):

    tag = None
    Meta = None

    @classmethod
    def from_xml(cls, xml):
        if cls.tag != xml.tag:
            raise ValueError(
                "Tag mismatch: expected %s but got %s" % (cls.tag, xml.tag))
        attrs = cls.get_attrs()
        inst = cls()
        used_attrs = []
        for name, attr in attrs:
            val = attr.from_xml(xml)
            if isinstance(attr, XmlAttr):
                used_attrs.append(attr.name)
            if val is not None:
                setattr(inst, name, val)
        for attr in list(xml.attrib.keys()):
            if attr not in used_attrs:
                raise ValueError("Unrecognized attribute: %s" % attr)
        return inst

    @classmethod
    def get_attrs(cls):
        if not hasattr(cls, '_attrs'):
            all_attrs = dir(cls.Meta)
            attrs = [(attr, getattr(cls.Meta, attr)) for attr in all_attrs
                     if isinstance(getattr(cls.Meta, attr), AttrDecl)]
            cls._attrs = attrs
        return cls._attrs

    def __str__(self):
        attrs = []
        for name, _ in self.__class__.get_attrs():
            attrs.append("%s=%s" % (name, str(getattr(self, name))))
        return self.__class__.__name__ + "(" + ", ".join(attrs) + ")"

    def __repr__(self):
        return str(self)
