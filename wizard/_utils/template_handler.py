import yaml


if __name__ == '__main__':

    import os
    file = os.path.join(os.getcwd(), '../../resources', 'hsi_workflow.yml')
    with open(file, 'r') as stream:
        try:
            print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
