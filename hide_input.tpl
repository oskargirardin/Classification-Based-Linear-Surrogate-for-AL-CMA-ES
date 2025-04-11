{% extends 'lab/index.html.j2' %}

{% block input %}
{% if 'hide_input' not in cell.metadata.get('tags', []) %}
    {{ super() }}
{% endif %}
{% endblock input %}

{% block output %}
{% if 'hide_output' not in cell.metadata.get('tags', []) %}
    {{ super() }}
{% endif %}
{% endblock output %}
