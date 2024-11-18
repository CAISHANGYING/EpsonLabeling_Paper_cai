import json
import numpy as np

class StandTool:

    def __init__(self, stand_tool_config : dict ) -> None:

        self._config = stand_tool_config

        self._tool_detail = {}

        self._load_tool()

    def _load_tool(self, tool_name : str = None ) -> None:

        if tool_name is None:

            tool_name = self._config['stand_tool_name']

        with open( self._config['parent_foleder'] + '/' + tool_name + '.json' ) as f:

            self._tool_detail = dict(json.load(f))

        self._name = self._tool_detail['name']
        self._shape = self._tool_detail['shape']
        self._realimage = np.array( self._tool_detail['real_image'], dtype='uint8')
        self._hole_detail = self._tool_detail['hole_detail']
        self._length_relationship = self._tool_detail['length_relationship']

        print( "load " + tool_name + " stand tool done.")

    def reload_tool(self, tool_name : str) -> None:

        self._load_tool( tool_name )

    def get_name(self) -> str:

        return self._name

    def get_shape(self) -> tuple:

        return self._shape[0], self._shape[1]

    def get_real_image(self) -> np.ndarray:

        return self._realimage

    def get_hole_detail(self) -> list:

        return self._hole_detail

    def get_hole_count(self) -> int:

        return len(self._hole_detail)

    def get_length_relationship(self) -> list:

        return self._length_relationship

    def get_length_relationship_convert(self) -> list:

        new_length_relationship = []

        for item in self._length_relationship:

            if not item[0].isdigit():

                relationship = [item[0][1:], [item[1][0][1:],item[1][1][1:]], item[2], item[3], item[4]]

                new_length_relationship.append(relationship)

            else:

                new_length_relationship.append(item)

        return new_length_relationship
