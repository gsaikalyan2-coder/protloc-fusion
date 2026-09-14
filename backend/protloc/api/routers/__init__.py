"""HTTP routers.

Each module here owns one group of endpoints and does four things: parse,
validate, delegate, return. If deleting this package would break the ability to
run a prediction from Python, the boundary is wrong — so nothing in it decides
anything.
"""
