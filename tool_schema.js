// this is the tool schema from https://github.com/nspady/google-calendar-mcp/blob/1bf7bbd1d52906b19686422d33004a2d3cb52e74/src/tools/registry.ts
// The fields that are going to be defaulted (for simplification) are commented

const timeMinSchema = z.string()
  .refine((val) => {
    const withTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/.test(val);
    const withoutTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(val);
    return withTimezone || withoutTimezone;
  }, "Must be ISO 8601 format: '2026-01-01T00:00:00'")
  .describe("Start time boundary. Preferred: '2024-01-01T00:00:00' (uses timeZone parameter or calendar timezone). Also accepts: '2024-01-01T00:00:00Z' or '2024-01-01T00:00:00-08:00'.")
  .optional();

const timeMaxSchema = z.string()
  .refine((val) => {
    const withTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/.test(val);
    const withoutTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(val);
    return withTimezone || withoutTimezone;
  }, "Must be ISO 8601 format: '2026-01-01T00:00:00'")
  .describe("End time boundary. Preferred: '2024-01-01T23:59:59' (uses timeZone parameter or calendar timezone). Also accepts: '2024-01-01T23:59:59Z' or '2024-01-01T23:59:59-08:00'.")
  .optional();

const timeZoneSchema = z.string().optional().describe(
  "Timezone as IANA Time Zone Database name (e.g., America/Los_Angeles). Takes priority over calendar's default timezone. Only used for timezone-naive datetime strings."
);

const fieldsSchema = z.preprocess(
  parseJsonStringArray,
  z.array(z.enum(ALLOWED_EVENT_FIELDS))
).optional().describe(
  "Optional array of additional event fields to retrieve. Available fields are strictly validated. Default fields (id, summary, start, end, status, htmlLink, location, attendees) are always included."
);

const calendarsToCheckSchema = z.preprocess(
  parseJsonStringArray,
  z.array(z.string())
).optional().describe(
  "List of calendar IDs to check for conflicts (defaults to just the target calendar)"
);

const recurrenceSchema = z.preprocess(
  parseJsonStringArray,
  z.array(z.string())
).optional().describe(
  "Recurrence rules in RFC5545 format (e.g., [\"RRULE:FREQ=WEEKLY;COUNT=5\"])"
);

const privateExtendedPropertySchema = z
  .array(z.string().regex(/^[^=]+=[^=]+$/, "Must be in key=value format"))
  .optional()
  .describe(
    "Filter by private extended properties (key=value). Matches events that have all specified properties."
  );

const sharedExtendedPropertySchema = z
  .array(z.string().regex(/^[^=]+=[^=]+$/, "Must be in key=value format"))
  .optional()
  .describe(
    "Filter by shared extended properties (key=value). Matches events that have all specified properties."
  );

// Single account schema - for write operations (create, update, delete)
const singleAccountSchema = z.string()
  .regex(/^[a-z0-9_-]{1,64}$/, "Account nickname must be 1-64 characters: lowercase letters, numbers, dashes, underscores only")
  .optional()
  .describe(
    "Account nickname to use for this operation (e.g., 'work', 'personal') - the friendly name you gave when connecting the account. Optional when only one account is connected - will auto-select the account with appropriate permissions. Use 'list-calendars' to see available accounts."
  );

// Account ID validation regex
const accountIdRegex = /^[a-z0-9_-]{1,64}$/;

// Multi-account schema - for read operations (list, search, get)
const multiAccountSchema = z.preprocess(
  parseAccountJsonString,
  z.union([
    z.string()
      .regex(accountIdRegex, "Account nickname must be 1-64 characters: lowercase letters, numbers, dashes, underscores only"),
    z.array(z.string()
      .regex(accountIdRegex, "Account nickname must be 1-64 characters: lowercase letters, numbers, dashes, underscores only"))
      .min(1, "At least one account nickname is required")
      .max(10, "Maximum 10 accounts allowed per request")
  ])
)
  .optional()
  .describe(
    "Account nickname(s) to query (e.g., 'work' or ['work', 'personal']) - the friendly names you gave when connecting accounts. Optional - if omitted, queries all connected accounts and merges results. Use 'list-calendars' to see available accounts."
  );

// Define all tool schemas with TypeScript inference
export const ToolSchemas = {
  'list-calendars': z.object({
    account: multiAccountSchema
  }),

  'list-events': z.object({
    account: multiAccountSchema,
    calendarId: z.union([
      z.string().describe(
        "Calendar identifier(s) to query. Accepts calendar IDs (e.g., 'primary', 'user@gmail.com') OR calendar names (e.g., 'Work', 'Personal'). Single calendar: 'primary'. Multiple calendars: array ['Work', 'Personal'] or JSON string '[\"Work\", \"Personal\"]'"
      ),
      z.array(z.string().min(1))
        .min(1, "At least one calendar ID is required")
        .max(50, "Maximum 50 calendars allowed per request")
        .refine(
          (arr) => new Set(arr).size === arr.length,
          "Duplicate calendar IDs are not allowed"
        )
        .describe("Array of calendar IDs to query events from (max 50, no duplicates)")
    ]),
    timeMin: timeMinSchema,
    timeMax: timeMaxSchema,
    timeZone: timeZoneSchema,
    fields: fieldsSchema,
    privateExtendedProperty: privateExtendedPropertySchema,
    sharedExtendedProperty: sharedExtendedPropertySchema
  }),
  
  'search-events': z.object({
    account: multiAccountSchema,
    calendarId: z.union([
      z.string().describe(
        "Calendar identifier(s) to search. Accepts calendar IDs (e.g., 'primary', 'user@gmail.com') OR calendar names (e.g., 'Work', 'Personal'). Single calendar: 'primary'. Multiple calendars: array ['Work', 'Personal'] or JSON string '[\"Work\", \"Personal\"]'"
      ),
      z.array(z.string())
    ]).transform((val) => {
      if (typeof val === 'string') {
        // Try to parse JSON array if it looks like one
        if (val.startsWith('[')) {
          try {
            const parsed = JSON.parse(val);
            if (Array.isArray(parsed)) return parsed;
          } catch { /* ignore */ }
        }
        return val;
      }
      return val;
    }).describe("Calendar identifier(s) to search. Accepts calendar IDs or names. Single or multiple calendars supported."),
    query: z.string().describe(
      "Free text search query (searches summary, description, location, attendees, etc.)"
    ),
    timeMin: z.string()
      .refine((val) => {
        const withTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/.test(val);
        const withoutTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(val);
        return withTimezone || withoutTimezone;
      }, "Must be ISO 8601 format: '2026-01-01T00:00:00'")
      .describe("Start time boundary. Preferred: '2024-01-01T00:00:00' (uses timeZone parameter or calendar timezone). Also accepts: '2024-01-01T00:00:00Z' or '2024-01-01T00:00:00-08:00'."),
    timeMax: z.string()
      .refine((val) => {
        const withTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/.test(val);
        const withoutTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(val);
        return withTimezone || withoutTimezone;
      }, "Must be ISO 8601 format: '2026-01-01T00:00:00'")
      .describe("End time boundary. Preferred: '2024-01-01T23:59:59' (uses timeZone parameter or calendar timezone). Also accepts: '2024-01-01T23:59:59Z' or '2024-01-01T23:59:59-08:00'."),
    timeZone: z.string().optional().describe(
      "Timezone as IANA Time Zone Database name (e.g., America/Los_Angeles). Takes priority over calendar's default timezone. Only used for timezone-naive datetime strings."
    ),
    fields: z.array(z.enum(ALLOWED_EVENT_FIELDS)).optional().describe(
      "Optional array of additional event fields to retrieve. Available fields are strictly validated. Default fields (id, summary, start, end, status, htmlLink, location, attendees) are always included."
    ),
    privateExtendedProperty: z
      .array(z.string().regex(/^[^=]+=[^=]+$/, "Must be in key=value format"))
      .optional()
      .describe(
        "Filter by private extended properties (key=value). Matches events that have all specified properties."
      ),
    sharedExtendedProperty: z
      .array(z.string().regex(/^[^=]+=[^=]+$/, "Must be in key=value format"))
      .optional()
      .describe(
        "Filter by shared extended properties (key=value). Matches events that have all specified properties."
      )
  }),
  
  'get-event': z.object({
    account: singleAccountSchema,
    calendarId: z.string().describe("ID of the calendar (use 'primary' for the main calendar)"),
    eventId: z.string().describe("ID of the event to retrieve"),
    fields: z.array(z.enum(ALLOWED_EVENT_FIELDS)).optional().describe(
      "Optional array of additional event fields to retrieve. Available fields are strictly validated. Default fields (id, summary, start, end, status, htmlLink, location, attendees) are always included."
    )
  }),

  'list-colors': z.object({
    account: singleAccountSchema,
  }),

  'create-event': z.object({
    account: singleAccountSchema,
    calendarId: z.string().describe("ID of the calendar (use 'primary' for the main calendar)"),
    eventId: z.string().optional().describe("Optional custom event ID (5-1024 characters, base32hex encoding: lowercase letters a-v and digits 0-9 only). If not provided, Google Calendar will generate one."),
    summary: z.string().describe("Title of the event"),
    description: z.string().optional().describe("Description/notes for the event"),
    start: z.string()
      .refine((val) => {
        const dateOnly = /^\d{4}-\d{2}-\d{2}$/.test(val);
        const withTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/.test(val);
        const withoutTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(val);
        return dateOnly || withTimezone || withoutTimezone;
      }, "Must be ISO 8601 format: '2025-01-01T10:00:00' for timed events or '2025-01-01' for all-day events")
      .describe("Event start time: '2025-01-01T10:00:00' for timed events or '2025-01-01' for all-day events. Also accepts Google Calendar API object format: {date: '2025-01-01'} or {dateTime: '2025-01-01T10:00:00', timeZone: 'America/Los_Angeles'}"),
    end: z.string()
      .refine((val) => {
        const dateOnly = /^\d{4}-\d{2}-\d{2}$/.test(val);
        const withTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/.test(val);
        const withoutTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(val);
        return dateOnly || withTimezone || withoutTimezone;
      }, "Must be ISO 8601 format: '2025-01-01T11:00:00' for timed events or '2025-01-02' for all-day events")
      .describe("Event end time: '2025-01-01T11:00:00' for timed events or '2025-01-02' for all-day events (exclusive). Also accepts Google Calendar API object format: {date: '2025-01-02'} or {dateTime: '2025-01-01T11:00:00', timeZone: 'America/Los_Angeles'}"),
    timeZone: z.string().optional().describe(
      "Timezone as IANA Time Zone Database name (e.g., America/Los_Angeles). Takes priority over calendar's default timezone. Only used for timezone-naive datetime strings."
    ),
    location: z.string().optional().describe("Location of the event"),
    attendees: z.array(z.object({
      email: z.string().email().describe("Email address of the attendee"),
      displayName: z.string().optional().describe("Display name of the attendee"),
      optional: z.boolean().optional().describe("Whether this is an optional attendee"),
      responseStatus: z.enum(["needsAction", "declined", "tentative", "accepted"]).optional().describe("Attendee's response status"),
      comment: z.string().optional().describe("Attendee's response comment"),
      additionalGuests: z.number().int().min(0).optional().describe("Number of additional guests the attendee is bringing")
    })).optional().describe("List of event attendees with their details"),
    colorId: z.string().optional().describe(
      "Color ID for the event (use list-colors to see available IDs)"
    ),
    reminders: z.object({
      useDefault: z.boolean().describe("Whether to use the default reminders"),
      overrides: z.array(z.object({
        method: z.enum(["email", "popup"]).default("popup").describe("Reminder method"),
        minutes: z.number().describe("Minutes before the event to trigger the reminder")
      }).partial({ method: true })).optional().describe("Custom reminders")
    }).describe("Reminder settings for the event").optional(),
    recurrence: recurrenceSchema,
    transparency: z.enum(["opaque", "transparent"]).optional().describe(
      "Whether the event blocks time on the calendar. 'opaque' means busy, 'transparent' means free."
    ),
    visibility: z.enum(["default", "public", "private", "confidential"]).optional().describe(
      "Visibility of the event. Use 'public' for public events, 'private' for private events visible to attendees."
    ),
    guestsCanInviteOthers: z.boolean().optional().describe(
      "Whether attendees can invite others to the event. Default is true."
    ),
    guestsCanModify: z.boolean().optional().describe(
      "Whether attendees can modify the event. Default is false."
    ),
    guestsCanSeeOtherGuests: z.boolean().optional().describe(
      "Whether attendees can see the list of other attendees. Default is true."
    ),
    anyoneCanAddSelf: z.boolean().optional().describe(
      "Whether anyone can add themselves to the event. Default is false."
    ),
    sendUpdates: z.enum(["all", "externalOnly", "none"]).optional().describe(
      "Whether to send notifications about the event creation. 'all' sends to all guests, 'externalOnly' to non-Google Calendar users only, 'none' sends no notifications."
    ),
    conferenceData: z.object({
      createRequest: z.object({
        requestId: z.string().describe("Client-generated unique ID for this request to ensure idempotency"),
        conferenceSolutionKey: z.object({
          type: z.enum(["hangoutsMeet", "eventHangout", "eventNamedHangout", "addOn"]).describe("Conference solution type")
        }).describe("Conference solution to create")
      }).describe("Request to generate a new conference")
    }).optional().describe(
      "Conference properties for the event. Use createRequest to add a new conference."
    ),
    extendedProperties: z.object({
      private: z.record(z.string()).optional().describe(
        "Properties private to the application. Keys can have max 44 chars, values max 1024 chars."
      ),
      shared: z.record(z.string()).optional().describe(
        "Properties visible to all attendees. Keys can have max 44 chars, values max 1024 chars."
      )
    }).optional().describe(
      "Extended properties for storing application-specific data. Max 300 properties totaling 32KB."
    ),
    attachments: z.array(z.object({
      fileUrl: z.string().describe("URL of the attached file"),
      title: z.string().optional().describe("Title of the attachment"),
      mimeType: z.string().optional().describe("MIME type of the attachment"),
      iconLink: z.string().optional().describe("URL of the icon for the attachment"),
      fileId: z.string().optional().describe("ID of the attached file in Google Drive")
    })).optional().describe(
      "File attachments for the event. Requires calendar to support attachments."
    ),
    source: z.object({
      url: z.string().describe("URL of the source"),
      title: z.string().describe("Title of the source")
    }).optional().describe(
      "Source of the event, such as a web page or email message."
    ),
    calendarsToCheck: calendarsToCheckSchema,
    duplicateSimilarityThreshold: z.number().min(0).max(1).optional().describe(
      "Threshold for duplicate detection (0-1, default: 0.7). Events with similarity above this are flagged as potential duplicates"
    ),
    allowDuplicates: z.boolean().optional().describe(
      "If true, allows creation even when exact duplicates are detected (similarity >= 0.95). Default is false which blocks duplicate creation"
    )
  }),
  
  'update-event': z.object({
    account: singleAccountSchema,
    calendarId: z.string().describe("ID of the calendar (use 'primary' for the main calendar)"),
    eventId: z.string().describe("ID of the event to update"),
    summary: z.string().optional().describe("Updated title of the event"),
    description: z.string().optional().describe("Updated description/notes"),
    start: z.string()
      .refine((val) => {
        const dateOnly = /^\d{4}-\d{2}-\d{2}$/.test(val);
        const withTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/.test(val);
        const withoutTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(val);
        return dateOnly || withTimezone || withoutTimezone;
      }, "Must be ISO 8601 format: '2025-01-01T10:00:00' for timed events or '2025-01-01' for all-day events")
      .describe("Updated start time: '2025-01-01T10:00:00' for timed events or '2025-01-01' for all-day events. Also accepts Google Calendar API object format: {date: '2025-01-01'} or {dateTime: '2025-01-01T10:00:00', timeZone: 'America/Los_Angeles'}")
      .optional(),
    end: z.string()
      .refine((val) => {
        const dateOnly = /^\d{4}-\d{2}-\d{2}$/.test(val);
        const withTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/.test(val);
        const withoutTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(val);
        return dateOnly || withTimezone || withoutTimezone;
      }, "Must be ISO 8601 format: '2025-01-01T11:00:00' for timed events or '2025-01-02' for all-day events")
      .describe("Updated end time: '2025-01-01T11:00:00' for timed events or '2025-01-02' for all-day events (exclusive). Also accepts Google Calendar API object format: {date: '2025-01-02'} or {dateTime: '2025-01-01T11:00:00', timeZone: 'America/Los_Angeles'}")
      .optional(),
    timeZone: z.string().optional().describe("Updated timezone as IANA Time Zone Database name. If not provided, uses the calendar's default timezone."),
    location: z.string().optional().describe("Updated location"),
    attendees: z.array(z.object({
      email: z.string().email().describe("Email address of the attendee")
    })).optional().describe("Updated attendee list"),
    colorId: z.string().optional().describe("Updated color ID"),
    reminders: z.object({
      useDefault: z.boolean().describe("Whether to use the default reminders"),
      overrides: z.array(z.object({
        method: z.enum(["email", "popup"]).default("popup").describe("Reminder method"),
        minutes: z.number().describe("Minutes before the event to trigger the reminder")
      }).partial({ method: true })).optional().describe("Custom reminders")
    }).describe("Reminder settings for the event").optional(),
    recurrence: recurrenceSchema,
    sendUpdates: z.enum(["all", "externalOnly", "none"]).default("all").describe(
      "Whether to send update notifications"
    ),
    modificationScope: z.enum(["thisAndFollowing", "all", "thisEventOnly"]).optional().describe(
      "Scope for recurring event modifications"
    ),
    originalStartTime: z.string()
      .refine((val) => {
        const withTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/.test(val);
        const withoutTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(val);
        return withTimezone || withoutTimezone;
      }, "Must be ISO 8601 format: '2026-01-01T00:00:00'")
      .describe("Original start time in the ISO 8601 format '2024-01-01T10:00:00'")
      .optional(),
    futureStartDate: z.string()
      .refine((val) => {
        const withTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/.test(val);
        const withoutTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(val);
        return withTimezone || withoutTimezone;
      }, "Must be ISO 8601 format: '2026-01-01T00:00:00'")
      .describe("Start date for future instances in the ISO 8601 format '2024-01-01T10:00:00'")
      .optional(),
    checkConflicts: z.boolean().optional().describe(
      "Whether to check for conflicts when updating (default: true when changing time)"
    ),
    calendarsToCheck: calendarsToCheckSchema,
    conferenceData: z.object({
      createRequest: z.object({
        requestId: z.string().describe("Client-generated unique ID for this request to ensure idempotency"),
        conferenceSolutionKey: z.object({
          type: z.enum(["hangoutsMeet", "eventHangout", "eventNamedHangout", "addOn"]).describe("Conference solution type")
        }).describe("Conference solution to create")
      }).describe("Request to generate a new conference for this event")
    }).optional().describe("Conference properties for the event. Used to add or update Google Meet links."),
    transparency: z.enum(["opaque", "transparent"]).optional().describe(
      "Whether the event blocks time on the calendar. 'opaque' means busy, 'transparent' means available"
    ),
    visibility: z.enum(["default", "public", "private", "confidential"]).optional().describe(
      "Visibility of the event"
    ),
    guestsCanInviteOthers: z.boolean().optional().describe(
      "Whether attendees other than the organizer can invite others"
    ),
    guestsCanModify: z.boolean().optional().describe(
      "Whether attendees other than the organizer can modify the event"
    ),
    guestsCanSeeOtherGuests: z.boolean().optional().describe(
      "Whether attendees other than the organizer can see who the event's attendees are"
    ),
    anyoneCanAddSelf: z.boolean().optional().describe(
      "Whether anyone can add themselves to the event"
    ),
    extendedProperties: z.object({
      private: z.record(z.string()).optional().describe("Properties that are private to the creator's app"),
      shared: z.record(z.string()).optional().describe("Properties that are shared between all apps")
    }).partial().optional().describe("Extended properties for the event"),
    attachments: z.array(z.object({
      fileUrl: z.string().url().describe("URL link to the attachment"),
      title: z.string().describe("Title of the attachment"),
      mimeType: z.string().optional().describe("MIME type of the attachment"),
      iconLink: z.string().optional().describe("URL link to the attachment's icon"),
      fileId: z.string().optional().describe("ID of the attached Google Drive file")
    })).optional().describe("File attachments for the event")
  }).refine(
    (data) => {
      // Require originalStartTime when modificationScope is 'thisEventOnly'
      if (data.modificationScope === 'thisEventOnly' && !data.originalStartTime) {
        return false;
      }
      return true;
    },
    {
      message: "originalStartTime is required when modificationScope is 'thisEventOnly'",
      path: ["originalStartTime"]
    }
  ).refine(
    (data) => {
      // Require futureStartDate when modificationScope is 'thisAndFollowing'
      if (data.modificationScope === 'thisAndFollowing' && !data.futureStartDate) {
        return false;
      }
      return true;
    },
    {
      message: "futureStartDate is required when modificationScope is 'thisAndFollowing'",
      path: ["futureStartDate"]
    }
  ).refine(
    (data) => {
      // Ensure futureStartDate is in the future when provided
      if (data.futureStartDate) {
        const futureDate = new Date(data.futureStartDate);
        const now = new Date();
        return futureDate > now;
      }
      return true;
    },
    {
      message: "futureStartDate must be in the future",
      path: ["futureStartDate"]
    }
  ),
  
  'delete-event': z.object({
    account: singleAccountSchema,
    calendarId: z.string().describe("ID of the calendar (use 'primary' for the main calendar)"),
    eventId: z.string().describe("ID of the event to delete"),
    sendUpdates: z.enum(["all", "externalOnly", "none"]).default("all").describe(
      "Whether to send cancellation notifications"
    )
  }),

  'get-freebusy': z.object({
    account: multiAccountSchema.describe(
      "Account nickname(s) to query from (e.g., 'work' or ['work', 'personal']). Optional - if omitted, queries from all connected accounts to maximize calendar accessibility."
    ),
    calendars: z.array(z.object({
      id: z.string().describe("ID of the calendar (use 'primary' for the main calendar)")
    })).describe(
      "List of calendars and/or groups to query for free/busy information"
    ),
    timeMin: z.string()
      .refine((val) => {
        const withTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/.test(val);
        const withoutTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(val);
        return withTimezone || withoutTimezone;
      }, "Must be ISO 8601 format: '2026-01-01T00:00:00'")
      .describe("Start time boundary. Preferred: '2024-01-01T00:00:00' (uses timeZone parameter or calendar timezone). Also accepts: '2024-01-01T00:00:00Z' or '2024-01-01T00:00:00-08:00'."),
    timeMax: z.string()
      .refine((val) => {
        const withTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/.test(val);
        const withoutTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(val);
        return withTimezone || withoutTimezone;
      }, "Must be ISO 8601 format: '2026-01-01T00:00:00'")
      .describe("End time boundary. Preferred: '2024-01-01T23:59:59' (uses timeZone parameter or calendar timezone). Also accepts: '2024-01-01T23:59:59Z' or '2024-01-01T23:59:59-08:00'."),
    timeZone: z.string().optional().describe("Timezone for the query"),
    groupExpansionMax: z.number().int().max(100).optional().describe(
      "Maximum number of calendars to expand per group (max 100)"
    ),
    calendarExpansionMax: z.number().int().max(50).optional().describe(
      "Maximum number of calendars to expand (max 50)"
    )
  }),
  
  'get-current-time': z.object({
    account: singleAccountSchema,
    timeZone: z.string().optional().describe(
      "Optional IANA timezone (e.g., 'America/Los_Angeles', 'Europe/London', 'UTC'). If not provided, uses the primary Google Calendar's default timezone."
    )
  }),

  'respond-to-event': z.object({
    calendarId: z.string().describe("ID of the calendar (use 'primary' for the main calendar)"),
    eventId: z.string().describe("ID of the event to respond to"),
    account: z.string().optional().describe(
      "Account nickname to use for this operation (e.g., 'work', 'personal'). Optional when only one account is connected - will auto-select the account with appropriate permissions. Use 'list-calendars' to see available accounts."
    ),
    response: z.enum(["accepted", "declined", "tentative", "needsAction"]).describe(
      "Your response to the event invitation: 'accepted' (accept), 'declined' (decline), 'tentative' (maybe), 'needsAction' (no response)"
    ),
    comment: z.string().optional().describe(
      "Optional message/note to include with your response (e.g., 'I have a conflict' when declining)"
    ),
    modificationScope: z.enum(["thisEventOnly", "all"]).optional().describe(
      "For recurring events: 'thisEventOnly' responds to just this instance, 'all' responds to all instances. Default is 'all'."
    ),
    originalStartTime: z.string()
      .refine((val) => {
        const withTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/.test(val);
        const withoutTimezone = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(val);
        return withTimezone || withoutTimezone;
      }, "Must be ISO 8601 format: '2025-01-01T10:00:00'")
      .describe("Original start time of the specific instance (required when modificationScope is 'thisEventOnly')")
      .optional(),
    sendUpdates: z.enum(["all", "externalOnly", "none"]).optional().describe(
      "Whether to send response notifications. 'all' sends to all guests, 'externalOnly' to non-Google Calendar users only, 'none' sends no notifications. Default is 'none'."
    )
  }).refine(
    (data) => {
      // Require originalStartTime when modificationScope is 'thisEventOnly'
      if (data.modificationScope === 'thisEventOnly' && !data.originalStartTime) {
        return false;
      }
      return true;
    },
    {
      message: "originalStartTime is required when modificationScope is 'thisEventOnly'",
      path: ["originalStartTime"]
    }
  )
} as const;