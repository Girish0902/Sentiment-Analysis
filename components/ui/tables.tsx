import type { ReactNode } from 'react';

export function DataTable({
  headers,
  rows,
  caption,
}: {
  headers: string[];
  rows: ReactNode[][];
  caption?: string;
}) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        {caption ? <caption>{caption}</caption> : null}
        <thead>
          <tr>
            {headers.map((header) => (
              <th key={header} scope="col">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>{row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
