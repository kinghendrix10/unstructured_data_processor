# unstructured_data_processor/output_formatter.py
import json
import csv
from typing import Dict, Any
import io

class OutputFormatter:
    def __init__(self):
        self.output_format = 'python_dict'  # Default to Python dictionary

    def set_output_format(self, format: str):
        self.output_format = format

    def format_output(self, data: Dict[str, Any]) -> Any:
        if self.output_format == 'python_dict':
            return data  # Return the data as-is
        elif self.output_format == 'json':
            return json.dumps(data, indent=2)
        elif self.output_format == 'csv':
            return self._to_csv(data)
        elif self.output_format == 'graphml':
            return self._to_graphml(data)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

    def _to_csv(self, data: Dict[str, Any]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write entities
        writer.writerow(['Entity ID', 'Entity Type', 'Entity Name', 'Metadata'])
        for entity in data['entities']:
            writer.writerow([entity['id'], entity['type'], entity['name'], json.dumps(entity['metadata'])])
        
        writer.writerow([])  # Empty row for separation
        
        # Write relationships
        writer.writerow(['Source ID', 'Target ID', 'Relationship Type', 'Metadata'])
        for rel in data['relationships']:
            writer.writerow([rel['source'], rel['target'], rel['type'], json.dumps(rel['metadata'])])
        
        return output.getvalue()

    def _to_graphml(self, data: Dict[str, Any]) -> str:
        # This is a basic GraphML implementation. For more complex graphs, consider using a library like NetworkX.
        output = io.StringIO()
        output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        output.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        output.write('  <graph id="G" edgedefault="directed">\n')
        
        # Write nodes (entities)
        for entity in data['entities']:
            output.write(f'    <node id="{entity["id"]}">\n')
            output.write(f'      <data key="type">{entity["type"]}</data>\n')
            output.write(f'      <data key="name">{entity["name"]}</data>\n')
            output.write(f'      <data key="metadata">{json.dumps(entity["metadata"])}</data>\n')
            output.write('    </node>\n')
        
        # Write edges (relationships)
        for rel in data['relationships']:
            output.write(f'    <edge source="{rel["source"]}" target="{rel["target"]}">\n')
            output.write(f'      <data key="type">{rel["type"]}</data>\n')
            output.write(f'      <data key="metadata">{json.dumps(rel["metadata"])}</data>\n')
            output.write('    </edge>\n')
        
        output.write('  </graph>\n')
        output.write('</graphml>')
        
        return output.getvalue()
