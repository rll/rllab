from __future__ import print_function
from __future__ import absolute_import

from rllab.misc import instrument
from nose2.tools import such


class TestClass(object):
    pass

with such.A("instrument") as it:
    @it.should("work")
    def test_concretize():
        it.assertEqual(instrument.concretize([5]), [5])
        it.assertEqual(instrument.concretize((5,)), (5,))
        fake_globals = dict(TestClass=TestClass)
        instrument.stub(fake_globals)
        modified = fake_globals["TestClass"]
        it.assertIsInstance(modified, instrument.StubClass)
        it.assertIsInstance(modified(), instrument.StubObject)
        it.assertEqual(instrument.concretize((5,)), (5,))
        it.assertIsInstance(instrument.concretize(modified()), TestClass)

it.createTests(globals())
