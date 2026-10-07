from tests.helpers.factories import make_generation, make_item


def item(i, **kw):
    return make_item(i, item_id=f"msvd:test:{i}", in_main=True, in_annotated=True, answers_ref=["dog", "puppy"], **kw)


def gen(gid, item_id, system_id, text, norm):
    return make_generation(0, gen_id=gid, item_id=item_id, system_id=system_id, output_text=text,
                           answer_norm=norm).model_dump()
