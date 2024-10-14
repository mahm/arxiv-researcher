def dict_to_xml_str(data: dict) -> str:
    xml_str = "<item>"
    for key, value in data.items():
        xml_str += f"<{key}>{value}</{key}>"
    xml_str += "</item>"
    return xml_str
