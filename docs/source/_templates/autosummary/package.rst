{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :show-inheritance:
   :noindex:

{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:

{% for item in modules %}
   {{ item }}
{% endfor %}
{% endif %}
