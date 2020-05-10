from configparser import ConfigParser


def save_config_file(section_names, config_dicts, file_path):
    config = ConfigParser()
    for i in range(len(section_names)):
        section_name = section_names[i]
        config.add_section(section_name)
        for key, value in config_dicts[i]:
            config.set(section_name, key, value)
    with open(file_path, 'w') as f:
        config.write(f)
    return


def read_config_file(file_path):
    section_names = []
    config_dicts = []
    config = ConfigParser()
    config.read(file_path)
    for section in config.sections():
        section_names.append(section)
        config_dicts.append(dict())
        for option in config.options(section):
            config_dicts[-1][option] = float(config.get(section, option))
    return section_names, config_dicts
