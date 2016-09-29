


from rllab.misc import instrument
from nose2.tools import such


class TestClass(object):
    @property
    def arr(self):
        return [1, 2, 3]

    @property
    def compound_arr(self):
        return [dict(a=1)]


with such.A("instrument") as it:
    @it.should
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


    @it.should
    def test_chained_call():
        fake_globals = dict(TestClass=TestClass)
        instrument.stub(fake_globals)
        modified = fake_globals["TestClass"]
        it.assertIsInstance(modified().arr[0], instrument.StubMethodCall)
        it.assertIsInstance(modified().compound_arr[0]["a"], instrument.StubMethodCall)
        it.assertEqual(instrument.concretize(modified().arr[0]), 1)


    @it.should
    def test_variant_generator():

        vg = instrument.VariantGenerator()
        vg.add("key1", [1, 2, 3])
        vg.add("key2", [True, False])
        vg.add("key3", lambda key2: [1] if key2 else [1, 2])
        it.assertEqual(len(vg.variants()), 9)

        class VG(instrument.VariantGenerator):

            @instrument.variant
            def key1(self):
                return [1, 2, 3]

            @instrument.variant
            def key2(self):
                yield True
                yield False

            @instrument.variant
            def key3(self, key2):
                if key2:
                    yield 1
                else:
                    yield 1
                    yield 2

        it.assertEqual(len(VG().variants()), 9)

it.createTests(globals())
