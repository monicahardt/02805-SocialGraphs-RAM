import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;

/** Builds an interactive HTML network from consecutive dialogue speakers. */
public final class InteractionGraph {
    private InteractionGraph() { }

    /** Writes a network where consecutive, different speakers in a chapter are connected. */
    public static void writeHtml(List<DialogueCsvReader.DialogueRecord> dialogue, Path outputPath)
            throws IOException {
        TreeSet<String> characters = new TreeSet<>();
        Map<String, Integer> weights = new TreeMap<>();
        Map<String, Integer> degrees = new TreeMap<>();
        String previousCharacter = null;
        String previousChapter = null;

        for (DialogueCsvReader.DialogueRecord row : dialogue) {
            if (row.character().equalsIgnoreCase("unknown")
                    || row.confidence().equalsIgnoreCase("low")
                    || row.confidence().equalsIgnoreCase("unattributed")) {
                // Do not make an artificial edge that jumps over an unattributed turn.
                previousCharacter = null;
                previousChapter = row.chapter();
                continue;
            }
            String character = row.character();
            characters.add(character);
            if (row.chapter().equals(previousChapter) && previousCharacter != null
                    && !previousCharacter.equals(character)) {
                String first = previousCharacter.compareTo(character) < 0 ? previousCharacter : character;
                String second = previousCharacter.compareTo(character) < 0 ? character : previousCharacter;
                String key = first + "\u0000" + second;
                if (weights.merge(key, 1, Integer::sum) == 1) {
                    degrees.merge(first, 1, Integer::sum);
                    degrees.merge(second, 1, Integer::sum);
                }
            }
            previousCharacter = character;
            previousChapter = row.chapter();
        }

        List<String> nodes = new ArrayList<>();
        for (String character : characters) {
            int degree = degrees.getOrDefault(character, 0);
            nodes.add("{id:" + json(character) + ", label:" + json(character)
                    + ", value:" + Math.max(1, degree)
                    + ", title:" + json(character + " — degree " + degree) + "}");
        }
        List<String> edges = new ArrayList<>();
        for (Map.Entry<String, Integer> edge : weights.entrySet()) {
            String[] pair = edge.getKey().split("\u0000", 2);
            int weight = edge.getValue();
            edges.add("{from:" + json(pair[0]) + ", to:" + json(pair[1]) + ", value:" + weight
                    + ", title:" + json(weight + " consecutive speaker transitions") + "}");
        }

        String title = dialogue.isEmpty() ? "Character dialogue network" : dialogue.get(0).book();
        String html = """
                <!doctype html>
                <html><head><meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>%s — Character Network</title>
                <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                <style>body{font:15px system-ui;margin:0;color:#222}header{padding:12px 20px}#network{height:calc(100vh - 75px);border-top:1px solid #ddd}</style>
                </head><body><header><strong>%s</strong> — character dialogue network<br>
                Each edge counts consecutive attributed turns by different speakers in the same chapter. Unknown and low-confidence turns are skipped; thicker edges mean more transitions.</header>
                <div id="network"></div><script>
                const nodes = new vis.DataSet([%s]);
                const edges = new vis.DataSet([%s]);
                new vis.Network(document.getElementById('network'), {nodes, edges}, {
                  nodes:{shape:'dot', font:{size:14}, scaling:{min:10,max:38}, color:{background:'#8ecae6',border:'#219ebc'}},
                  edges:{smooth:{type:'dynamic'}, color:{color:'#777',highlight:'#e76f51'}, scaling:{min:1,max:8}},
                  interaction:{hover:true}, physics:{solver:'forceAtlas2Based', forceAtlas2Based:{
                    gravitationalConstant:-80, centralGravity:0.015, springLength:130,
                    springConstant:0.08, avoidOverlap:0.8
                  }, stabilization:{iterations:800}}
                });
                </script></body></html>
                """.formatted(html(title), html(title), String.join(",", nodes), String.join(",", edges));
        Path parent = outputPath.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
        Files.writeString(outputPath, html, StandardCharsets.UTF_8);
    }

    private static String json(String value) {
        StringBuilder escaped = new StringBuilder("\"");
        for (char c : value.toCharArray()) {
            switch (c) {
                case '"' -> escaped.append("\\\"");
                case '\\' -> escaped.append("\\\\");
                case '\n' -> escaped.append("\\n");
                case '\r' -> escaped.append("\\r");
                case '\t' -> escaped.append("\\t");
                default -> {
                    if (c < 0x20) escaped.append(String.format("\\u%04x", (int) c));
                    else escaped.append(c);
                }
            }
        }
        return escaped.append('"').toString();
    }

    private static String html(String value) {
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\"", "&quot;").replace("'", "&#39;");
    }
}
