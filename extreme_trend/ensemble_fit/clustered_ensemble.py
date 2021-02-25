
"""Instead of creating one big group, with all the ensemble together,
and assuming a common temporal trend to all the group.

We could create smaller groups, with an importance proportional to the number of GCM/RCM couples considered
in the group.
For instance, we could group them by GCM, or group them by RCM.
Or we could try to find a metric to group them together.

This is the idea of finding of sweet spot between:
-only independent fits with few assumptions
-one common fit with too much assumption


it links with the idea of "climate model subset".

Generally people try to find one model subset,
the idea here, would be to find group of model subsets
"""